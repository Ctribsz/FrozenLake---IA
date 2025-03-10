import gymnasium as gym
import numpy as np

# esta libreria practicamente ya tiene su propia implementacion del juego de frozen lake.
# el juego trata de llegar desde un inicio hasta el final sin caer en lugares donde hayan agujeros con agua.
# el juego es una cuadricula de 4x4, osea tenemos 16 estados (del 0 al 15)
# el objetivo es hacer una politica (osea como una formula para que le agente pueda tomar decisiones y llegar al final)
# el personaje se puede mover hacia arriba, abajo y a los lados (como en el lab anterior del laberinto)
# pero el asunto se pone complicado cuando se pone la variable is_slippery, lo que hace que haya una probabilidad
# de que al intentar ir a cualquier lado, vaya a otro, osea ya no es determinista el movimiento

# en canvas hay un enlace a un ejemplo, pero en ese ejemplo el piso no es slippery
# https://aleksandarhaber.com/policy-iteration-algorithm-in-python-and-tests-with-frozen-lake-openai-gym-environment-reinforcement-learning-tutorial/
# otra buena referencia, pero relacionada a qlearning
# https://towardsdatascience.com/q-learning-for-beginners-2837b777741/

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
#environment.close()

for _ in range(episodes):
    state, r_info = environment.reset()
    finished = False

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

        # If we have a reward, it means that our outcome is a success
        if reward:
          print("llegamos al final :) ")
        
        #print(qtable)

print()
print('Tabla post entrenamiento:')
print(qtable)