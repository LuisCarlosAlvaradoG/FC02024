#!/usr/bin/env python3
"""
Script para convertir PDFs a Markdown - Optimizado para letra manuscrita
Usa EasyOCR que es mucho mejor para documentos escaneados con escritura a mano
"""

import os
from pathlib import Path
from markitdown import MarkItDown
import easyocr
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

def mejorar_imagen(imagen_pil):
    """
    Mejora la imagen para mejor OCR
    """
    try:
        imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
        gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
        
        # CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contraste = clahe.apply(gris)
        
        # Binarización adaptativa
        binaria = cv2.adaptiveThreshold(
            contraste, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return Image.fromarray(binaria)
    
    except Exception as e:
        print(f"   ⚠️  Error al mejorar imagen: {e}")
        return imagen_pil

def extraer_texto_easyocr(pdf_path):
    """
    Extrae texto usando EasyOCR (mucho mejor para manuscritos)
    
    Args:
        pdf_path: Ruta del PDF
    
    Returns:
        str: Texto extraído
    """
    try:
        print("   🔍 Inicializando EasyOCR (primera vez descarga modelos)...")
        
        # Inicializar lector con soporte para español e inglés
        reader = easyocr.Reader(['es', 'en'], gpu=False)
        
        print("   📄 Convirtiendo PDF a imágenes...")
        imagenes = convert_from_path(pdf_path, dpi=300)
        
        texto_completo = ""
        
        for num_pagina, imagen in enumerate(imagenes, 1):
            print(f"   📖 Leyendo página {num_pagina}... (OCR para manuscritos)", end=" ")
            
            # Mejorar imagen
            imagen_mejorada = mejorar_imagen(imagen)
            
            # Convertir a array para EasyOCR
            imagen_array = np.array(imagen_mejorada)
            
            # Detectar texto
            resultados = reader.readtext(imagen_array, detail=0)
            
            # Unir texto
            texto_pagina = '\n'.join(resultados)
            
            if texto_pagina.strip():
                texto_completo += f"## Página {num_pagina}\n\n{texto_pagina}\n\n"
            
            print("✓")
        
        return texto_completo if texto_completo.strip() else None
    
    except Exception as e:
        print(f"   ❌ Error en EasyOCR: {e}")
        return None

def convertir_pdf(pdf_path):
    """
    Convierte PDF usando EasyOCR (principal) o MarkItDown (fallback)
    """
    
    try:
        print("   📝 Detectando tipo de documento...")
        
        # Primero intentar EasyOCR (mejor para manuscritos)
        print("   ℹ️  Usando EasyOCR (especializado en manuscritos)")
        texto = extraer_texto_easyocr(pdf_path)
        
        if texto and len(texto.strip()) > 100:
            return texto
        
        # Si no funcionó, intentar MarkItDown
        print("   🔄 Intentando MarkItDown como fallback...")
        md = MarkItDown()
        resultado = md.convert(str(pdf_path))
        
        if len(resultado.text_content) > 100:
            return resultado.text_content
        
        return None
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def convertir_pdfs():
    """
    Convierte todos los PDFs en Img_Apuntes a Markdown
    """
    
    carpeta = "Img_Apuntes"
    
    if not os.path.exists(carpeta):
        print(f"❌ Carpeta '{carpeta}' no encontrada")
        return
    
    archivos_pdf = sorted(list(Path(carpeta).glob("*.pdf")))
    
    if not archivos_pdf:
        print(f"⚠️  No hay PDFs en '{carpeta}'")
        return
    
    print(f"📂 Encontrados {len(archivos_pdf)} PDF(s)\n")
    print("=" * 60)
    
    convertidos = 0
    ignorados = 0
    errores = 0
    
    for pdf_file in archivos_pdf:
        nombre_base = pdf_file.stem
        archivo_md = pdf_file.parent / f"{nombre_base}.md"
        
        if archivo_md.exists():
            print(f"⏭️  IGNORADO: {archivo_md.name}")
            ignorados += 1
            continue
        
        print(f"📄 Procesando: {pdf_file.name}")
        texto = convertir_pdf(str(pdf_file))
        
        if texto and len(texto.strip()) > 50:
            try:
                with open(archivo_md, "w", encoding="utf-8") as f:
                    f.write(texto)
                
                print(f"✅ CREADO: {archivo_md.name}\n")
                convertidos += 1
            except Exception as e:
                print(f"❌ Error al guardar: {e}\n")
                errores += 1
        else:
            print(f"❌ No se pudo extraer texto\n")
            errores += 1
    
    print("=" * 60)
    print(f"\n📊 RESUMEN: ✅ {convertidos} | ⏭️  {ignorados} | ❌ {errores}\n")

if __name__ == "__main__":
    print("🚀 Convertidor PDF → Markdown (EasyOCR para manuscritos)\n")
    convertir_pdfs()