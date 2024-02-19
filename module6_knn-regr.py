
import numpy as np

class KNNRegressor:
    def __init__(self, k):
        self.k = k
        self.points = None

    def add_points(self, points):
        self.points = np.array(points)

    def predict(self, x):
        if self.points is None or len(self.points) == 0:
            raise ValueError("No points have been added to the regressor.")
        if self.k > len(self.points):
            raise ValueError("k cannot be greater than the number of points.")

        # Calculate distances from the point to predict (x) to all other points
        distances = np.sqrt((self.points[:, 0] - x) ** 2)

        # Get indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:self.k]

        # Calculate the average y value of the k nearest neighbors
        average_y = np.mean(self.points[nearest_indices, 1])

        return average_y

def main():
    try:
        N = int(input("Enter the number of points (N): "))
        if N <= 0:
            raise ValueError("N must be a positive integer.")

        k = int(input("Enter the number of nearest neighbors (k): "))
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        points = []
        print("Enter the points (x, y):")
        for _ in range(N):
            x = float(input("x: "))
            y = float(input("y: "))
            points.append((x, y))

        X = float(input("Enter the value of X to predict Y: "))

        knn = KNNRegressor(k)
        knn.add_points(points)
        Y = knn.predict(X)

        print(f"The predicted value of Y for X = {X} using {k}-NN Regression is: {Y}")

    except ValueError as e:
        print(e)
    except Exception as e:
        print("An unexpected error occurred:", e)

if __name__ == "__main__":
    main()
