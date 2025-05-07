#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

// Graph class representing an undirected graph using adjacency list representation
class Graph {
private:
    int numVertices;          // Number of vertices
    vector<vector<int>> adj;  // Adjacency list

public:
    Graph(int vertices) : numVertices(vertices), adj(vertices) {}

    // Add an edge between two vertices
    void addEdge(int src, int dest) {
        adj[src].push_back(dest);
        adj[dest].push_back(src);
    }

    // View the graph
    void viewGraph() {
        cout << "Graph:\n";
        for (int i = 0; i < numVertices; i++) {
            cout << "Vertex " << i << " -> ";
            for (int neighbor : adj[i]) {
                cout << neighbor << " ";
            }
            cout << endl;
        }
    }

    // Perform Breadth First Search (BFS) in parallel
    // Perform Breadth First Search (BFS) in parallel with execution time and thread logs
void bfs(int startVertex) {
    double start_time = omp_get_wtime();

    vector<bool> visited(numVertices, false);
    queue<int> q;
    vector<int> bfsOrder;

    visited[startVertex] = true;
    q.push(startVertex);

    while (!q.empty()) {
        int currentVertex = q.front();
        q.pop();
        bfsOrder.push_back(currentVertex);

        // Parallel processing of neighbors
        #pragma omp parallel for
        for (int i = 0; i < adj[currentVertex].size(); i++) {
            int neighbor = adj[currentVertex][i];
            printf("Thread %d is processing neighbor %d of vertex %d\n", omp_get_thread_num(), neighbor, currentVertex);

            // Use a critical section to safely update shared visited and queue
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    double end_time = omp_get_wtime();

    // Display BFS order
    cout << "\nBreadth First Search (BFS) Order: ";
    for (int v : bfsOrder) {
        cout << v << " ";
    }
    cout << "\nParallel BFS completed in " << (end_time - start_time) * 1000 << " milliseconds\n";
}

// Perform Depth First Search (DFS) in parallel with execution time and thread logs
void dfs(int startVertex) {
    double start_time = omp_get_wtime();

    vector<bool> visited(numVertices, false);
    stack<int> s;
    vector<int> dfsOrder;

    visited[startVertex] = true;
    s.push(startVertex);

    while (!s.empty()) {
        int currentVertex = s.top();
        s.pop();
        dfsOrder.push_back(currentVertex);

        // Parallel processing of neighbors
        #pragma omp parallel for
        for (int i = 0; i < adj[currentVertex].size(); i++) {
            int neighbor = adj[currentVertex][i];
            printf("Thread %d is processing neighbor %d of vertex %d\n", omp_get_thread_num(), neighbor, currentVertex);

            // Use a critical section to safely update shared visited and stack
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    s.push(neighbor);
                }
            }
        }
    }

    double end_time = omp_get_wtime();

    // Display DFS order
    cout << "\nDepth First Search (DFS) Order: ";
    for (int v : dfsOrder) {
        cout << v << " ";
    }
    cout << "\nParallel DFS completed in " << (end_time - start_time) * 1000 << " milliseconds\n";
}

};

int main() {
    int numVertices;
    cout << "Enter the number of vertices in the graph: ";
    cin >> numVertices;

    // Create a graph with the specified number of vertices
    Graph graph(numVertices);

    int numEdges;
    cout << "Enter the number of edges in the graph: ";
    cin >> numEdges;

    cout << "Enter the edges (source destination):\n";
    for (int i = 0; i < numEdges; i++) {
        int src, dest;
        cin >> src >> dest;
        graph.addEdge(src, dest);
    }

    // View the graph
    graph.viewGraph();

    int startVertex;
    cout << "Enter the starting vertex for BFS and DFS: ";
    cin >> startVertex;

    cout << "Breadth First Search (BFS): ";
    graph.bfs(startVertex);
    cout << endl;

    cout << "Depth First Search (DFS): ";
    graph.dfs(startVertex);
    cout << endl;

    return 0;
}