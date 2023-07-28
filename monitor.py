
def initialize_file(filename):
    """
    Initializes the CSV file with the header.
    :param filename: Name of the file to initialize.
    """
    header = ['Iteration', 'Objective function', 'Gradient Norm']

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

    # Define a callback function
def store_data(xk, filename):
    global iteration
    objective_value = driver.fun(xk)  # Evaluate the objective function value
    gradient_norm = np.linalg.norm(driver.grad(xk))  # Calculate the gradient norm
    data = [iteration, objective_value, gradient_norm]
    iteration += 1

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\t')
        writer.writerow(data)