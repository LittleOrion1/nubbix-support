from pydantic import BaseModel, Field, field_validator
from typing import Literal

Category = Literal["billing", "technical", "account", "other"]

class SupportResponse(BaseModel):
    # Corregido: model_config es reservado, uso model_name
    model_name: str = Field(..., min_length=5, max_length=15)
    answer: str = Field(..., min_length=10, max_length=100)
    
    # Corregido: faltaba el tipo de dato (float) antes de Field
    confidence: float = Field(..., ge=0.9, le=1.0)
    category: Category
    actions: list[str] = Field(..., min_length=1, max_length=3) 

    @field_validator("answer")
    @classmethod
    def validar_respuesta(cls, value: str) -> str:
        """Valida que la respuesta del LLM no contenga inyecciones o comandos peligrosos."""
        minuscula = value.lower() 
        valores_prohibidos = [
            "<script>", 
            "drop table", 
            "rm -rf", 
            "eval(", 
            "exec("
        ]
        if any(token in minuscula for token in valores_prohibidos):
            raise ValueError("Respuesta inválida: contiene contenido prohibido.")
        
        return value

    @field_validator("actions")
    @classmethod
    def limpiar_acciones(cls, value: list[str]) -> list[str]:
        #Limpia los espacios en blanco de las acciones y rechaza elementos vacíos
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned: 
            raise ValueError("Debe haber al menos una acción válida.")
        return cleaned
    
