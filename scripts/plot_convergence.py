import pandas as pd
import matplotlib.pyplot as plt

# expects data/convergence.csv with columns: iter,residual
df = pd.read_csv("data/convergence.csv")
plt.plot(df['iter'], df['residual'], marker='o')
plt.xlabel("Newton iteration")
plt.ylabel("||F||")
plt.title("Newton convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/convergence.png")
print("Saved data/convergence.png")
