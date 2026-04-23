"""
Aggregate APIRouter for version 1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import decision_routes, explain_routes, risk_routes, simulation_routes, suggestion_routes

api_router = APIRouter()
api_router.include_router(risk_routes.router)
api_router.include_router(decision_routes.router)
api_router.include_router(simulation_routes.router)
api_router.include_router(explain_routes.router)
api_router.include_router(suggestion_routes.router)
