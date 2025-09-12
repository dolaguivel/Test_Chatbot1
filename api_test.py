import os
from typing import Optional, List
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

DATA_PATH = os.getenv("DATA_PATH", "Titanic-Dataset.xlsx")

def load_dataframe() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        if DATA_PATH.lower().endswith(".xlsx"):
            df = pd.read_excel(DATA_PATH, engine="openpyxl")
        else:
            df = pd.read_csv(DATA_PATH)
    else:
        # fallback: busca un csv clásico si no existe el xlsx
        for alt in ["titanic.csv", "titanic_dataset.csv", "Titanic-Dataset.csv"]:
            if os.path.exists(alt):
                return pd.read_csv(alt)
        raise FileNotFoundError(
            f"No se encontró el dataset en {DATA_PATH}. "
            "Configura la variable de entorno DATA_PATH o sube el archivo."
        )
    return df

df = load_dataframe()
# guardamos nombres y tipos
COLUMNS = list(df.columns)
DTYPES = {c: str(df[c].dtype) for c in COLUMNS}

app = FastAPI(
    title="Titanic API",
    description="API para explorar el Titanic Dataset (xlsx/csv) con filtros, paginación y estadísticas.",
    version="1.0.0",
)

# CORS abierto para facilitar consumo desde webapps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def apply_filters(df_in: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Filtros genéricos por querystring.
    Reglas:
      - col=value           -> igualdad (string o numérico)
      - col__contains=txt   -> contiene (string, case-insensitive)
      - col__gte=10         -> >=
      - col__lte=20         -> <=
      - col__ne=value       -> !=
    """
    df_out = df_in.copy()
    for k, v in params.items():
        # ignora parámetros reservados
        if k in {"limit", "offset", "order_by", "order_dir", "select", "search", "groupby", "format"}:
            continue
        if "__" in k:
            col, op = k.split("__", 1)
        else:
            col, op = k, "eq"

        if col not in df_out.columns:
            # parámetro que no es columna: lo ignoramos (no lanzamos error para ser amable)
            continue

        series = df_out[col]
        if op == "contains":
            df_out = df_out[series.astype(str).str.contains(str(v), case=False, na=False)]
        elif op == "gte":
            df_out = df_out[pd.to_numeric(series, errors="coerce") >= pd.to_numeric(v, errors="coerce")]
        elif op == "lte":
            df_out = df_out[pd.to_numeric(series, errors="coerce") <= pd.to_numeric(v, errors="coerce")]
        elif op == "ne":
            df_out = df_out[series.astype(str) != str(v)]
        else:  # eq
            # intenta numérico; si no, compara como string
            num_series = pd.to_numeric(series, errors="coerce")
            num_v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if pd.notna(num_v) and num_series.notna().any():
                df_out = df_out[num_series == num_v]
            else:
                df_out = df_out[series.astype(str) == str(v)]
    return df_out

@app.get("/")
def root():
    return {
        "name": "Titanic API",
        "rows": len(df),
        "columns": COLUMNS,
        "endpoints": {
            "/columns": "Lista de columnas",
            "/schema": "Tipos de datos por columna",
            "/count": "Cantidad de filas (con filtros)",
            "/data": "Datos con filtros/paginación/orden/selección",
            "/unique/{column}": "Valores únicos de una columna",
            "/describe": "Estadística descriptiva",
            "/stats": "Estadísticas por grupo (groupby)",
            "/download": "Descarga filtrada en CSV o JSON",
        },
        "filter_syntax": {
            "eq": "col=value",
            "ne": "col__ne=value",
            "contains": "col__contains=texto",
            "gte": "col__gte=10",
            "lte": "col__lte=20"
        }
    }

@app.get("/columns")
def get_columns():
    return COLUMNS

@app.get("/schema")
def get_schema():
    return DTYPES

@app.get("/count")
async def count(request: Request):
    params = dict(request.query_params)
    filtered = apply_filters(df, params)
    return {"count": len(filtered)}

@app.get("/data")
async def get_data(
    request: Request,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: Optional[str] = None,
    order_dir: Optional[str] = Query("asc", regex="^(asc|desc)$"),
    select: Optional[str] = None,  # columnas separadas x coma
):
    params = dict(request.query_params)
    # filtra
    filtered = apply_filters(df, params)

    # selecciona columnas si se pide
    if select:
        cols = [c.strip() for c in select.split(",") if c.strip()]
        for c in cols:
            if c not in filtered.columns:
                raise HTTPException(status_code=400, detail=f"Columna desconocida: {c}")
        filtered = filtered[cols]

    # orden
    if order_by:
        if order_by not in filtered.columns:
            raise HTTPException(status_code=400, detail=f"Columna de orden desconocida: {order_by}")
        filtered = filtered.sort_values(order_by, ascending=(order_dir == "asc"))

    total = len(filtered)
    data = filtered.iloc[offset: offset + limit].to_dict(orient="records")
    return {
        "meta": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "returned": len(data)
        },
        "data": data
    }

@app.get("/unique/{column}")
def unique_values(column: str, limit: int = Query(2000, ge=1, le=10000)):
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columna desconocida: {column}")
    vals = pd.Series(df[column].unique()).dropna().head(limit).tolist()
    return {"column": column, "unique": vals, "count": len(vals)}

@app.get("/describe")
def describe(select: Optional[str] = None):
    if select:
        cols = [c.strip() for c in select.split(",") if c.strip()]
        for c in cols:
            if c not in df.columns:
                raise HTTPException(status_code=400, detail=f"Columna desconocida: {c}")
        d = df[cols].describe(include="all").transpose().fillna("").reset_index()
    else:
        d = df.describe(include="all").transpose().fillna("").reset_index()
    return d.to_dict(orient="records")

@app.get("/stats")
def stats(
    groupby: str = Query(..., description="Columna por la que agrupar, p. ej. Sex o Pclass"),
    agg: List[str] = Query(["count"], description="Funciones de agregación: count, mean, median, sum, min, max"),
    select: Optional[str] = Query(None, description="Columnas numéricas a agregar, separadas por coma"),
):
    if groupby not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columna de grupo desconocida: {groupby}")
    if select:
        cols = [c.strip() for c in select.split(",") if c.strip()]
        for c in cols:
            if c not in df.columns:
                raise HTTPException(status_code=400, detail=f"Columna desconocida: {c}")
        to_agg = {c: agg for c in cols}
    else:
        # auto: todas numéricas
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            raise HTTPException(status_code=400, detail="No hay columnas numéricas para agregar.")
        to_agg = {c: agg for c in num_cols}

    out = df.groupby(groupby).agg(to_agg)
    # aplanar MultiIndex de columnas
    out.columns = ["__".join([c for c in tup if c]) for tup in out.columns.values]
    out = out.reset_index()
    return out.to_dict(orient="records")

@app.get("/download")
async def download(request: Request, format: str = Query("csv", regex="^(csv|json)$")):
    params = dict(request.query_params)
    filtered = apply_filters(df, params)
    if format == "json":
        return filtered.to_dict(orient="records")
    # CSV como texto
    return filtered.to_csv(index=False)

