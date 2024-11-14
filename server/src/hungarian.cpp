#include "hungarian.hpp"

double HungarianAlgorithm::Solve(const std::vector<std::vector<double>>& costMatrix, std::vector<int>& assignment) {
    int nOfRows = costMatrix.size();
    int nOfColumns = costMatrix[0].size();
    
    double *distMatrix = new double[nOfRows * nOfColumns];
    for (int i = 0; i < nOfRows; i++)
        for (int j = 0; j < nOfColumns; j++)
            distMatrix[i + nOfRows * j] = costMatrix[i][j];

    assignment.resize(nOfRows, -1);
    double cost = 0.0;
    
    assignmentoptimal(&assignment[0], &cost, distMatrix, nOfRows, nOfColumns);
    
    delete[] distMatrix;
    return cost;
}

void HungarianAlgorithm::assignmentoptimal(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns) {
    bool *starMatrix = new bool[nOfRows * nOfColumns];
    bool *newStarMatrix = new bool[nOfRows * nOfColumns];
    bool *primeMatrix = new bool[nOfRows * nOfColumns];
    bool *coveredColumns = new bool[nOfColumns];
    bool *coveredRows = new bool[nOfRows];
    int minDim = std::min(nOfRows, nOfColumns);
    
    std::fill(starMatrix, starMatrix + nOfRows * nOfColumns, false);
    std::fill(newStarMatrix, newStarMatrix + nOfRows * nOfColumns, false);
    std::fill(primeMatrix, primeMatrix + nOfRows * nOfColumns, false);
    std::fill(coveredColumns, coveredColumns + nOfColumns, false);
    std::fill(coveredRows, coveredRows + nOfRows, false);
    
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    
    delete[] starMatrix;
    delete[] newStarMatrix;
    delete[] primeMatrix;
    delete[] coveredColumns;
    delete[] coveredRows;
}

void HungarianAlgorithm::step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
    // Star zeros in each row
    for (int row = 0; row < nOfRows; row++) {
        for (int col = 0; col < nOfColumns; col++) {
            if (distMatrix[row + nOfRows * col] == 0 && !coveredColumns[col] && !coveredRows[row]) {
                starMatrix[row + nOfRows * col] = true;  // Star the zero
                coveredColumns[col] = true;  // Cover the column
                coveredRows[row] = true;  // Cover the row
                break;  // Move to the next row
            }
        }
    }

    // Clear the covered rows and columns for the next steps
    std::fill(coveredRows, coveredRows + nOfRows, false);
    std::fill(coveredColumns, coveredColumns + nOfColumns, false);

    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
    // Cover columns containing starred zeros
    for (int row = 0; row < nOfRows; row++) {
        for (int col = 0; col < nOfColumns; col++) {
            if (starMatrix[row + nOfRows * col]) {
                coveredColumns[col] = true;
            }
        }
    }

    // Check if enough columns are covered to form an optimal assignment
    int numCoveredColumns = 0;
    for (int col = 0; col < nOfColumns; col++) {
        if (coveredColumns[col]) {
            numCoveredColumns++;
        }
    }

    if (numCoveredColumns >= minDim) {
        // We have found an optimal assignment
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    } else {
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

void HungarianAlgorithm::step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zeroFound = true;
    while (zeroFound) {
        zeroFound = false;
        for (int row = 0; row < nOfRows; row++) {
            if (!coveredRows[row]) {
                for (int col = 0; col < nOfColumns; col++) {
                    if (!coveredColumns[col] && distMatrix[row + nOfRows * col] == 0) {
                        primeMatrix[row + nOfRows * col] = true;  // Prime the zero
                        // If there's a starred zero in the row, cover the row and uncover the column
                        bool starInRow = false;
                        for (int j = 0; j < nOfColumns; j++) {
                            if (starMatrix[row + nOfRows * j]) {
                                starInRow = true;
                                coveredRows[row] = true;
                                coveredColumns[j] = false;
                                break;
                            }
                        }
                        if (!starInRow) {
                            // If no star in row, go to step 4
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        zeroFound = true;
                        break;
                    }
                }
            }
        }
    }

    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col) {
    bool done = false;
    int zRow = row;
    int zCol = col;
    int starCol = -1;

    while (!done) {
        starCol = -1;
        for (int j = 0; j < nOfColumns; j++) {
            if (starMatrix[zRow + nOfRows * j]) {
                starCol = j;
                break;
            }
        }

        if (starCol != -1) {
            newStarMatrix[zRow + nOfRows * zCol] = true;
            primeMatrix[zRow + nOfRows * zCol] = false;
            zRow = -1;

            for (int i = 0; i < nOfRows; i++) {
                if (primeMatrix[i + nOfRows * starCol]) {
                    zRow = i;
                    break;
                }
            }

            if (zRow == -1) {
                done = true;
            } else {
                zCol = starCol;
            }
        } else {
            done = true;
        }
    }

    // Update starMatrix to reflect the augmenting path
    for (int i = 0; i < nOfRows * nOfColumns; i++) {
        if (newStarMatrix[i]) {
            starMatrix[i] = true;
        }
        primeMatrix[i] = false;  // Clear primes
    }

    // Clear covers
    std::fill(coveredRows, coveredRows + nOfRows, false);
    std::fill(coveredColumns, coveredColumns + nOfColumns, false);

    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
    // Find the smallest uncovered value
    double minUncoveredValue = std::numeric_limits<double>::max();
    for (int row = 0; row < nOfRows; row++) {
        if (!coveredRows[row]) {
            for (int col = 0; col < nOfColumns; col++) {
                if (!coveredColumns[col]) {
                    minUncoveredValue = std::min(minUncoveredValue, distMatrix[row + nOfRows * col]);
                }
            }
        }
    }

    // Subtract the minimum from all uncovered elements
    for (int row = 0; row < nOfRows; row++) {
        if (!coveredRows[row]) {
            for (int col = 0; col < nOfColumns; col++) {
                if (!coveredColumns[col]) {
                    distMatrix[row + nOfRows * col] -= minUncoveredValue;
                }
            }
        }
    }

    // Add the minimum to all covered rows
    for (int col = 0; col < nOfColumns; col++) {
        if (coveredColumns[col]) {
            for (int row = 0; row < nOfRows; row++) {
                distMatrix[row + nOfRows * col] += minUncoveredValue;
            }
        }
    }

    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns) {
    // Clear the current assignment
    for (int row = 0; row < nOfRows; row++) {
        assignment[row] = -1;  // Initialize assignment to -1 (no assignment)
    }

    // Loop over the starMatrix to assign rows to columns
    for (int row = 0; row < nOfRows; row++) {
        for (int col = 0; col < nOfColumns; col++) {
            if (starMatrix[row + nOfRows * col]) {  // If there's a star at (row, col)
                assignment[row] = col;  // Assign the row to this column
                break;  // Move to the next row once the assignment is found
            }
        }
    }
}
