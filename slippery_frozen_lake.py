import gymnasium as gym

# esta libreria practicamente ya tiene su propia implementacion del juego de frozen lake.
# el juego trata de llegar desde un inicio hasta el final sin caer en lugares donde hayan agujeros con agua.
# el juego es una cuadricula de 4x4, osea tenemos 16 estados (del 0 al 15)
# el objetivo es hacer una politica (osea como una formula para que le agente pueda tomar decisiones y llegar al final)
# el personaje se puede mover hacia arriba, abajo y a los lados (como en el lab anterior del laberinto)
# pero el asunto se pone complicado cuando se pone la variable is_slippery, lo que hace que haya una probabilidad
# de que al intentar ir a cualquier lado, vaya a otro, osea ya no es determinista el movimiento

# en canvas hay un enlace a un ejemplo, pero en ese ejemplo el piso no es slippery
# https://aleksandarhaber.com/policy-iteration-algorithm-in-python-and-tests-with-frozen-lake-openai-gym-environment-reinforcement-learning-tutorial/

# definicion del ambiente
environment = gym.make('FrozenLake-v1', render_mode="human", is_slippery = True)
initial_state, info = environment.reset()

print("Estado inicial:", initial_state)
print("Información adicional:", info)

# prueba de una accion
action = environment.action_space.sample() # un movimiento al azar

# luego de hacer una accion se puede obtener toda la informacion del ambiente, recompensa y demas
new_state, reward, finished, trunacted, info = environment.step(action)


print("Nuevo estado:", new_state)
print("Recompensa:", reward)

# por lo que entiendo el episodio termina o bien cuando cae a un agujero o llega al final
print("¿Episodio terminado?:", finished)

environment.render()

#environment.close()