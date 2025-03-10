import gymnasium as gym
import numpy as np

# definicion del ambiente
# el render mode se puede cambiar, human es el que se ve mejor visualmente pero es mas lento
environment = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery = True)
initial_state, info = environment.reset()

print("Estado inicial:", initial_state)
print("Información adicional:", info)

# prueba de una accion
action = environment.action_space.sample() # un movimiento al azar

# luego de hacer una accion se puede obtener toda la informacion del ambiente, recompensa y demas
new_state, reward, finished, truncated, info = environment.step(action)

print("Nuevo estado:", new_state)
print("Recompensa:", reward)

# por lo que entiendo el episodio termina o bien cuando cae a un agujero o llega al final
print("¿Episodio terminado?:", finished)
environment.render()

# estrategia: utilizar q learning, porque? es un metodo de aprendizaje por refurezo util en situaciones no discretas
# lo que pasa es que ayuda a aprender mediante las interacciones con el ambiente. El q learning toma un estado y unaccion
# y retorna la recompensa esperada (optima)

# para hacer esto va a ser util tener una tabla, que ayude a representar los estados y posibles acciones
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))
print('Q-table')
print(qtable)
# cada una de las filas de la matriz es un estado y cada elemento de la lista es el valor de movernos hacia una direccion
# siendo 1 lo mas alto posible y 0 lo mas bajo. 1 solo se obtiene al llegar a la meta y 0 al caer en un agujero

# la estrategia puede ser hacer que el personaje se mueva aleatoriamente hasta que caiga en un agujero o llegue al final
# y a medida que hacemos estas pruebas se actualiza la matriz para conocer los caminos optimos

# hiperparametros
episodes = 1000        # epoc
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor

# Contadores para medir el desempeño
# la cantidad de epoc exitosos
# y el total de los pasos, esto al final es para que podamos saber el promedio de pasos
# para encontrar la salida
successful_episodes = 0
total_steps_success = 0

for _ in range(episodes):
    state, r_info = environment.reset()
    finished = False

    steps = 0

    while not finished:
        # en un estado, elegir la accion con mayor reward
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])
        # si todos son 0 agarrar cualquiera
        else:
          action = environment.action_space.sample()

        new_state, reward, finished, truncated, info = environment.step(action)
        new_state, reward, finished, info

        #   q(s,a)
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # nuevo estado
        state = new_state
        steps += 1

        environment.render()

        # si hay un reward de 1 llegamos a la meta, sino quiere decir que no llegamos y caimos en aun agujero
        if reward == 1:
            successful_episodes += 1
            total_steps_success += steps
            print(f"\n\nLlegamos a la meta en {steps} pasos :)\n")
        elif finished:
            print("No llegamos a la meta :( ", end = "")

environment.close()
print()
print('Tabla post entrenamiento:')
print(qtable)

if successful_episodes > 0:
    average_steps = total_steps_success / successful_episodes
    print(f"\nPromedio de pasos para llegar a la meta: {average_steps:.2f}")
else:
    print("\nNo hubo episodios exitosos.")

# Al probar varias veces el promedio de pasos usualmente termina siendo entre 10 y 15
# pero
# pero aun se podria mejorar de 2 formas, ajustando los hiperparametros
# o ajustando el qlearning para usar "Epsilon greedy algorithm"
