import sys
from src.model_predict import predict_fit

print("=== TESTE INTERATIVO DO MODELO BASELINE ===")
print("Digite as informações abaixo para avaliar o Fit:\n")

vaga = input("Cole a Descrição da Vaga (JD): ")
curriculo = input("\nCole o Texto do Currículo: ")

if vaga.strip() and curriculo.strip():
    try:
        resultado = predict_fit(curriculo, vaga)
        print("\n" + "="*30)
        print("RESULTADO DA AVALIAÇÃO:")
        print(f"Status: {resultado['status']}")
        print(f"Confiança: {resultado['score_confianca'] * 100:.2f}%")
        print("="*30)
    except Exception as e:
        print(f"\n[ERRO] Não foi possível classificar: {e}")
else:
    print("\n[AVISO] Você precisa preencher ambos os campos para testar.")