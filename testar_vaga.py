from src.model_predict import predict_fit

# 1. Colamos o texto da vaga dentro de uma string tripla (bloco de texto)
vaga_real = """
Experiência prática com Visão Computacional, incluindo classificação, detecção, segmentação ou reconhecimento de imagens.
Conhecimento em frameworks de Deep Learning, como PyTorch ou TensorFlow/Keras.
Experiência com bibliotecas de processamento de imagens, como OpenCV, Pillow, scikit-image ou similares.
Proficiência em Python e SQL.
Experiência com desenvolvimento, validação e implantação de modelos de Machine Learning.
Conhecimento em estatística aplicada e análise exploratória de dados.
Experiência com ambientes de dados em cloud ou plataformas analíticas, como Azure, AWS, GCP ou Databricks.
Conhecimento em IA Generativa, LLMs, agentes ou aplicações similares.
Capacidade de comunicação e interação com áreas técnicas e de negócio.
Inglês para leitura de documentação técnica.

Diferenciais:
Experiência com Databricks em projetos de dados e Inteligência Artificial.
Vivência com modelos pré-treinados de Visão Computacional, como YOLO, Detectron2, Vision Transformers (ViT) ou similares.
Conhecimento em MLOps, incluindo versionamento, monitoramento e ciclo de vida de modelos.
"""

# 2. Um currículo fictício bem aderente para testar o 'Fit'
curriculo_teste = """
Junior Data Scientist focado em Visão Computacional e IA Generativa. 
Proficiente em Python, SQL e no desenvolvimento de modelos de Machine Learning e Deep Learning utilizando PyTorch e Scikit-Learn.
Experiência prática no processamento de imagens com OpenCV e modelagem preditiva. 
Conhecimento em ferramentas analíticas no Databricks e Azure. Boa comunicação técnica.
"""

print("=== EXECUTANDO MODELO BASELINE NA VAGA REAL ===")
try:
    resultado = predict_fit(curriculo_teste, vaga_real)
    print("\n" + "="*40)
    print("RESULTADO DO CASO DE TESTE:")
    print(f"Status do Match: {resultado['status']}")
    print(f"Confiança Matemática: {resultado['score_confianca'] * 100:.2f}%")
    print("="*40)
except Exception as e:
    print(f"\n[ERRO] Falha ao processar predição: {e}")