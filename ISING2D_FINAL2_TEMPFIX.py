import numba
import numpy as np
from math import log, e, sqrt
import matplotlib.pyplot as plt
from PIL import Image

kT_critical = 2 / log(1 + sqrt(2), e)

@numba.jit(nopython=True)
def expos(kT):
    beta = 1.0 / kT
    return np.array([np.exp(8.0 * beta), np.exp(4.0 * beta), 1.0, np.exp(-4.0 * beta), np.exp(-8.0 * beta)])

@numba.jit(nopython=True)
def _update(x, i, j, ex, use_expos, kT):
    n, m = x.shape
    dE = 2 * x[i, j] * (
        x[(i-1)%n, j] + x[(i+1)%n, j] + x[i, (j-1)%m] + x[i, (j+1)%m]
    )
    if use_expos:
        if dE <= 0 or ex[dE//4 + 2] > np.random.random():
            x[i, j] *= -1
    else:
        if dE <= 0 or np.exp(-dE / kT) > np.random.random():
            x[i, j] *= -1

@numba.jit(nopython=True)
def update(x, ex, use_expos, kT):
    n, m = x.shape
    for i in range(n):
        for j in range(0, m, 2):  # Colunas pares
            _update(x, i, j, ex, use_expos, kT)
    for i in range(n):
        for j in range(1, m, 2):  # Colunas ímpares
            _update(x, i, j, ex, use_expos, kT)

@numba.jit(nopython=True)
def magnetization(x):
    return np.sum(x) / x.size

@numba.jit(nopython=True)
def energy(x):
    """Calcula a energia total da configuração de spins."""
    n, m = x.shape
    E = 0.0
    for i in range(n):
        for j in range(m):
            E -= x[i, j] * (x[i, (j+1) % m] + x[(i+1) % n, j])
    return E

@numba.jit(nopython=True)
def calculate_specific_heat_and_susceptibility_per_box(energies, magnetizations, beta, n_boxes, Nterm, n, m):
    # Usar dados somente após o passo de termalização
    energies = energies[Nterm:]
    magnetizations = magnetizations[Nterm:]
    N_spins = n * m  # Número total de spins na rede
    N = len(energies)
    m_steps = N // n_boxes  # Número de passos por caixa

    Cv_values = np.zeros(n_boxes)
    Chi_values = np.zeros(n_boxes)
    Cv_err_values = np.zeros(n_boxes)
    Chi_err_values = np.zeros(n_boxes)
    energia_por_spin_values = np.zeros(n_boxes)
    magnetizacao_por_spin_values = np.zeros(n_boxes)

    # Calcular os valores de C_v, \chi, energia por spin e magnetização por spin para cada caixa
    for i in range(n_boxes):
        start = i * m_steps
        end = start + m_steps

        E_avg = np.mean(energies[start:end])
        E2_avg = np.mean(energies[start:end]**2)

        M_avg = np.mean(magnetizations[start:end])
        M2_avg = np.mean(magnetizations[start:end]**2)

        Cv_values[i] = (beta**2 / N_spins) * (E2_avg - E_avg**2)
        Chi_values[i] = (beta / N_spins) * (M2_avg - M_avg**2)

        energia_por_spin_values[i] = E_avg / N_spins
        magnetizacao_por_spin_values[i] = M_avg / N_spins

    # Calcular os erros para C_v e \chi de forma correta
    for i in range(n_boxes):
        start = i * m_steps
        end = start + m_steps

        Cv_err_values[i] = np.sqrt(np.sum((Cv_values - Cv_values[i])**2) / (n_boxes * (n_boxes - 1)))
        Chi_err_values[i] = np.sqrt(np.sum((Chi_values - Chi_values[i])**2) / (n_boxes * (n_boxes - 1)))

    results = np.zeros((n_boxes, 7))
    for i in range(n_boxes):
        results[i, 0] = i + 1
        results[i, 1] = Cv_values[i]
        results[i, 2] = Cv_err_values[i]
        results[i, 3] = Chi_values[i]
        results[i, 4] = Chi_err_values[i]
        results[i, 5] = energia_por_spin_values[i]
        results[i, 6] = magnetizacao_por_spin_values[i]

    return results


def display_spin_field(x):
    return Image.fromarray(np.uint8((x + 1) * 0.5 * 255))

def simulate_ising(num_steps, temp, n, m, use_expos=True, Nterm=400, n_boxes=100):
    magnetizations = np.zeros(num_steps)
    energies = np.zeros(num_steps)
    x = np.random.choice([-1, 1], size=(n, m)).astype('i1')
    kT = temp * kT_critical
    beta = 1.0 / kT
    ex = expos(kT) if use_expos else np.array([kT])  # Calcular as exponenciais ou passar kT

    step_indices = np.linspace(0, num_steps - 1, 6).astype(int)
    images = []

    for step in range(num_steps):
        update(x, ex, use_expos, kT)
        magnetizations[step] = magnetization(x)
        energies[step] = energy(x)

        if step in step_indices:
            img = display_spin_field(x)
            images.append(img)

    # Salvar os dados em um arquivo .txt
#    with open(f'ising_simulation_data{num_steps}_{temp}.txt', 'w') as f:
#        f.write("Step\tEnergy\tMagnetization\n")
#        for step in range(num_steps):
#            f.write(f"{step}\t{energies[step]}\t{magnetizations[step]}\n")
    
    # Salvar os dados em um arquivo .txt
    with open(f'ativ2_L{n}_boxes{n_boxes}_t{temp}.txt', 'w') as f:
        f.write("Box\tCv\tCv_err\tChi\tChi_err\tEnergia_por_Spin\tMagnetizacao_por_Spin\n")

        # Calcular calor específico, susceptibilidade e salvar por caixas
        results = calculate_specific_heat_and_susceptibility_per_box(energies, magnetizations, beta, n_boxes, Nterm, n, m)
        np.savetxt(f, results, fmt="%d\t%.13f\t%.13f\t%.13f\t%.13f\t%.13f\t%.13f")

    # Calcular energia por spin e magnetização por spin geral (média de todas as caixas)
    energia_por_spin = np.mean(energies[Nterm:]) / (n * m)
    magnetizacao_por_spin = np.mean(magnetizations[Nterm:]) / (n * m)

    return magnetizations, images, energia_por_spin, magnetizacao_por_spin

if __name__ == '__main__':
    n, m = 300, 300
    num_steps = 100400  # Escolhido para (num_steps - Nterm) ser divisível por n_boxes
    temp = 0.4  # Temperatura fixa para o plot ao longo do tempo
    Nterm = 400  # Passo de termalização
    n_boxes = 1000  # Número de caixas para método das caixas
    
    # Rodar simulação com ou sem expos
    magnetizations, images, energia_por_spin, magnetizacao_por_spin = simulate_ising(
        num_steps, temp, n, m, use_expos=True, Nterm=Nterm, n_boxes=n_boxes
    )
    
    # Plotar magnetização média em função do tempo
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(num_steps), magnetizations, 'o-')
    plt.xlabel('Passos de Tempo')
    plt.ylabel('Magnetização Média')
    plt.title(f'Magnetização Média em Função do Tempo (T/Tc = {temp})')
    plt.grid(True)
    plt.show()
    
    # Exibir os resultados de energia por spin e magnetização por spin
    print(f"Energia por Spin (media): {energia_por_spin}")
    print(f"Magnetizacao por Spin (media): {magnetizacao_por_spin}")
    
    # Salvar ou mostrar imagens geradas
#    for idx, img in enumerate(images):
#        img.save(f'ising2d_step_{idx}.png')  # Salvar se desejado
