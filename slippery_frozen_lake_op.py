# En esta implementacion se buscan los parametros optimos y despues una Q - table final usando el algorimo e - greedy
import gymnasium as gym
import numpy as np
import random

def train_q_learning(alpha, gamma, epsilon_decay, num_episodes=2000, max_steps=100):
    """
    Entrena un agente Q-learning en FrozenLake con is_slippery=True usando la estrategia epsilon-greedy.
    Retorna la Q-table obtenida.
    """
    env = gym.make('FrozenLake-v1', is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    epsilon = 1.0
    epsilon_min = 0.01

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    env.close()
    return Q

def evaluate_q_table(Q, num_eval_episodes=1000, max_steps=100):
    """
    Evalúa la Q-table en un entorno idéntico (FrozenLake con is_slippery=True) usando la política greedy.
    Retorna el número promedio de pasos y la tasa de éxito (episodios en los que se alcanzó la meta).
    """
    env = gym.make('FrozenLake-v1', is_slippery=True)
    total_steps = 0
    total_successes = 0

    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = np.argmax(Q[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            state = next_state
            done = terminated or truncated
            if done and reward == 1:
                total_successes += 1
        total_steps += steps

    env.close()
    avg_steps = total_steps / num_eval_episodes
    success_rate = total_successes / num_eval_episodes
    return avg_steps, success_rate

def search_optimal_hyperparameters(alpha_range, gamma_range, epsilon_decay_range, train_episodes=2000, eval_episodes=1000, max_steps=100):
    """
    Recorre las combinaciones de hiperparámetros (alpha, gamma, epsilon_decay) y evalúa la Q-table obtenida.
    Se escoge la combinación que maximice la tasa de éxito y, en caso de empate, la que tenga menor promedio de pasos.
    Retorna la mejor combinación, la tasa de éxito, el promedio de pasos y la lista de resultados.
    """
    best_params = None
    best_success_rate = -1.0
    best_avg_steps = float('inf')
    results = []

    for alpha in alpha_range:
        for gamma in gamma_range:
            for epsilon_decay in epsilon_decay_range:
                Q = train_q_learning(alpha, gamma, epsilon_decay, num_episodes=train_episodes, max_steps=max_steps)
                avg_steps, success_rate = evaluate_q_table(Q, num_eval_episodes=eval_episodes, max_steps=max_steps)
                results.append((alpha, gamma, epsilon_decay, success_rate, avg_steps))
                print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon Decay: {epsilon_decay} --> Tasa de éxito: {success_rate*100:.2f}%, Promedio de pasos: {avg_steps:.2f}")
                if success_rate > best_success_rate or (success_rate == best_success_rate and avg_steps < best_avg_steps):
                    best_success_rate = success_rate
                    best_avg_steps = avg_steps
                    best_params = (alpha, gamma, epsilon_decay)

    return best_params, best_success_rate, best_avg_steps, results

def demo_agent(Q, num_demo_episodes=10, max_steps=100):
    """
    Ejecuta una demostración visual del agente usando la Q-table obtenida.
    Se utiliza el entorno FrozenLake con is_slippery=True y render_mode="human"
    para mostrar cómo el agente se mueve.
    """
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="human")

    for episode in range(num_demo_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        print(f"\n--- Episodio {episode+1} ---")
        while not done and steps < max_steps:
            # En este modo, render_mode="human" ya muestra la simulación en una ventana
            action = np.argmax(Q[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            steps += 1
            done = terminated or truncated
        print(f"Episodio terminado en {steps} pasos. Recompensa obtenida: {reward}")

    env.close()

# Rango de hiperparámetros a evaluar
alphas = [0.1, 0.5, 0.8]
gammas = [0.8, 0.9, 0.95]
epsilon_decays = [0.99, 0.995, 0.999]

# Búsqueda de hiperparámetros óptimos
best_params, best_success_rate, best_avg_steps, results = search_optimal_hyperparameters(
    alphas, gammas, epsilon_decays, train_episodes=2000, eval_episodes=1000, max_steps=100
)

print("\nMejores hiperparámetros encontrados:")
print(f"Alpha: {best_params[0]}, Gamma: {best_params[1]}, Epsilon Decay: {best_params[2]}")
print(f"Tasa de éxito: {best_success_rate*100:.2f}%")
print(f"Promedio de pasos: {best_avg_steps:.2f}")

# Entrenar de nuevo usando los mejores hiperparámetros para obtener la Q-table final
best_alpha, best_gamma, best_epsilon_decay = best_params
Q_final = train_q_learning(best_alpha, best_gamma, best_epsilon_decay, num_episodes=2000, max_steps=100)

# Demostración visual del agente (render_mode="human")
demo_agent(Q_final, num_demo_episodes=10, max_steps=100)
