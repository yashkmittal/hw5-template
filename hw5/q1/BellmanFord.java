import java.util.ArrayList;

public class BellmanFord extends LLP {
    // You may be given inputs with negative weights, but no negative cost cycles
    // The graph may be directed, the weight of the edge from i to j is adjMatrix[i][j]

    int[][] adjMatrix;
    ArrayList<ArrayList<Integer>> pre;
    int[] min;
    int source_v;

    public BellmanFord(int[][] adjMatrix, int source) {
        super(adjMatrix.length);
        this.adjMatrix = adjMatrix;
        this.pre = new ArrayList<ArrayList<Integer>>();
        this.min = new int[adjMatrix.length];
        this.source_v = source;
        for(int i = 0; i < adjMatrix.length; i++){
            pre.add(new ArrayList<Integer>());
        }
        for(int i = 0; i < adjMatrix.length; i++){
            for(int j = 0; j < adjMatrix.length; j++){
                if (j == source_v) {
                    continue;
                }
                if(adjMatrix[i][j] != 0){
                    pre.get(j).add(i);
                }
            }
        }
    }

    @Override
    public void init(int j) {
        G[j] = (j == source_v) ? 0 : Integer.MAX_VALUE;
        min[j] = (j == source_v) ? 0 : Integer.MAX_VALUE;
    }

    @Override
    public boolean forbidden(int j) {
        boolean forbidden = false;
        for(Integer node: pre.get(j)){
            int pos_num = G[node] + adjMatrix[node][j];
            if(pos_num < G[node]) pos_num = Integer.MAX_VALUE;
            if(min[j] > pos_num){
                min[j] = pos_num;
            }
            if(G[j] > pos_num){
                forbidden = true;
            }
        }
        return forbidden;
    }

    @Override
    public void advance(int j) {
        G[j] = min[j];
    }

    // This method will be called after solve()
    public int[] getSolution() {
        // Return the vector of shortest path costs from source to each vertex
        // If a vertex is not connected to the source then its cost is Integer.MAX_VALUE
        return G;
    }
}
