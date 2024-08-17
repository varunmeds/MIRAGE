import pandas as pd
import matplotlib.pyplot as plt

chunk_size = 1000  # Adjust chunk size as needed
chunks = []

with open('query_times.txt', 'r') as file:
    chunk = []
    for line in file:
        try:
            number = float(line.strip())
            chunk.append(number)
            if len(chunk) >= chunk_size:
                chunks.append(chunk)
                chunk = []
        except ValueError:
            print(f"Skipping line: '{line.strip()}' (could not convert to float)")

    # Append any remaining data
    if chunk:
        chunks.append(chunk)

# Flatten the list of chunks
cleaned_numbers = [num for sublist in chunks for num in sublist]

# Convert cleaned numbers to a Pandas DataFrame
numbers_df = pd.DataFrame(cleaned_numbers, columns=['Value'])

# Plotting using Pandas
numbers_df.plot(y='Value', use_index=True, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Numbers from Text File')

# Save the plot as an image file
plt.savefig('numbers_plot_pandas.png')

# Display the graph
plt.grid(True)
plt.show()
