import math

def erlang_c(A, N):
    """
    Calcula la probabilidad de espera con Erlang-C.
    A: Carga de trabajo en Erlangs.
    N: Número de agentes.
    """
    numerador = (A**N / math.factorial(N)) / sum((A**i / math.factorial(i)) for i in range(N + 1))
    denominador = 1 - (numerador)
    return numerador / denominador if denominador > 0 else 1

def calcular_agentes(volumen_llamadas, tmo, nivel_servicio=0.8):
    """
    Calcula el número óptimo de agentes con Erlang-C.
    volumen_llamadas: llamadas por hora.
    tmo: tiempo medio de operación (en minutos).
    nivel_servicio: porcentaje de llamadas atendidas sin espera.
    """
    A = (volumen_llamadas * tmo) / 60  # Carga de trabajo en Erlangs
    N = math.ceil(A)  # Número inicial de agentes

    while True:
        E = erlang_c(A, N)
        probabilidad_atencion = 1 - E
        
        if probabilidad_atencion >= nivel_servicio:
            return N  # Devuelve la cantidad óptima de agentes
        N += 1
