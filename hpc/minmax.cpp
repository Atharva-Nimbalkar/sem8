#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Parallel reduction to find the minimum value
int parallelMin(const vector<int>& vec) {
    int min_val = vec[0];
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] < min_val)
            min_val = vec[i];
    }
    return min_val;
}


// Parallel reduction to find the maximum value
int parallelMax(const vector<int>& vec) {
    int max_val = vec[0];
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] > max_val)
            max_val = vec[i];
    }
    return max_val;
}


// Parallel reduction to calculate the sum
int parallelSum(const vector<int>& vec) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}


// Average using parallel sum
float parallelAverage(const vector<int>& vec) {
    int sum = parallelSum(vec);
    return static_cast<float>(sum) / vec.size();
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> vec(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> vec[i];
    }

    auto start = high_resolution_clock::now();

    int min_val = parallelMin(vec);
    cout << "Minimum value: " << min_val << endl;

    int max_val = parallelMax(vec);
    cout << "Maximum value: " << max_val << endl;

    int sum = parallelSum(vec);
    cout << "Sum of values: " << sum << endl;

    float avg = parallelAverage(vec);
    cout << "Average of values: " << avg << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\nExecution time: " << duration.count() << " microseconds" << endl;


    return 0;
}
