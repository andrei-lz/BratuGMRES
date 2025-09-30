import pandas as pd
import matplotlib.pyplot as plt

# expects data/strong_scaling.csv with columns: procs,time
df = pd.read_csv("data/strong_scaling.csv")
p1 = df['time'][0]
df['speedup'] = p1 / df['time']
df['efficiency'] = df['speedup'] / df['procs']

plt.figure()
plt.plot(df['procs'], df['speedup'], marker='o')
plt.xlabel("MPI ranks")
plt.ylabel("Speedup")
plt.title("Strong scaling speedup")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/strong_speedup.png")

plt.figure()
plt.plot(df['procs'], df['efficiency'], marker='o')
plt.xlabel("MPI ranks")
plt.ylabel("Parallel efficiency")
plt.title("Strong scaling efficiency")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/strong_efficiency.png")
print("Saved data/strong_speedup.png and data/strong_efficiency.png")
