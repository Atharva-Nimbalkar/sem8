#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
using namespace std;

void min_reduction(vector<int>& arr) {
    int min_value = INT_MAX;
    double start = omp_get_wtime();

    cout << "\n--- Parallel Execution for Minimum ---\n";
    #pragma omp parallel for reduction(min: min_value)
    for (int i = 0; i < arr.size(); i++) {
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num() << " processing index " << i << " (value: " << arr[i] << ")\n";
        }
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }

    double end = omp_get_wtime();
    cout << "Minimum value: " << min_value << endl;
    cout << "Time taken: " << (end - start)*1000 << " milliseconds\n";
}

void max_reduction(vector<int>& arr) {
    int max_value = INT_MIN;
    double start = omp_get_wtime();

    cout << "\n--- Parallel Execution for Maximum ---\n";
    #pragma omp parallel for reduction(max: max_value)
    for (int i = 0; i < arr.size(); i++) {
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num() << " processing index " << i << " (value: " << arr[i] << ")\n";
        }
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }

    double end = omp_get_wtime();
    cout << "Maximum value: " << max_value << endl;
    cout << "Time taken: " << (end - start)*1000 << " milliseconds\n";
}

void sum_reduction(vector<int>& arr) {
    int sum = 0;
    double start = omp_get_wtime();

    cout << "\n--- Parallel Execution for Sum ---\n";
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < arr.size(); i++) {
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num() << " adding index " << i << " (value: " << arr[i] << ")\n";
        }
        sum += arr[i];
    }

    double end = omp_get_wtime();
    cout << "Sum: " << sum << endl;
    cout << "Time taken: " << (end - start)*1000 << " milliseconds\n";
}

void average_reduction(vector<int>& arr) {
    int sum = 0;
    double start = omp_get_wtime();

    cout << "\n--- Parallel Execution for Average ---\n";
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < arr.size(); i++) {
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num() << " processing index " << i << " (value: " << arr[i] << ")\n";
        }
        sum += arr[i];
    }

    double end = omp_get_wtime();
    cout << "Average: " << (double)sum / arr.size() << endl;
    cout << "Time taken: " << (end - start)*1000 << "milliseconds\n";
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter " << n << " elements: ";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    min_reduction(arr);
    max_reduction(arr);
    sum_reduction(arr);
    average_reduction(arr);

    return 0;
}