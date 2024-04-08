import java.util.ArrayList;

public class Johnsons extends LLP {
    // You may be given inputs with negative weights, but no negative cost cycles
    // The graph may be directed, the weight of the edge from i to j is adjMatrix[i][j]
    int[][] adjMatrix;
    ArrayList<ArrayList<Integer>> pre;
    int[] max;


    public Johnsons(int[][] adjMatrix) {
        super(adjMatrix.length);
        this.adjMatrix = adjMatrix;
        this.pre = new ArrayList<ArrayList<Integer>>();
        this.max = new int[adjMatrix.length];
        for(int i = 0; i < adjMatrix.length; i++){
            pre.add(new ArrayList<Integer>());
        }
        for(int i = 0; i < adjMatrix.length; i++){
            for(int j = 0; j < adjMatrix.length; j++){
                if(adjMatrix[i][j] != 0){
                    pre.get(j).add(i);
                }
            }
        }
    }

    @Override
    public void init(int j) {
        G[j] = 0;
        max[j] = Integer.MIN_VALUE;
    }

    @Override
    public boolean forbidden(int j) {
        for(Integer node: pre.get(j)){
            if(G[node] - adjMatrix[node][j] > max[j]){
                max[j] = G[node] - adjMatrix[node][j];
            }
        }
        if(G[j] < max[j]){
            return true;
        }
        return false;
        
    }

    @Override
    public void advance(int j) {
        G[j] = max[j];
    }

    // This method will be called after solve()
    public int[] getSolution() {
        // Return the minimum price vector from Johnson's algorithm
        return G;
    }
}