public class ConnectedComponents extends LLP {

    int[][] adjMatrix;
    int[] max;

    public ConnectedComponents(int[][] adjMatrix) {
        super(adjMatrix.length);
        this.adjMatrix = adjMatrix;
        this.max = new int[adjMatrix.length];
    }

    @Override
    public void init(int j) {
        G[j] = j;
        max[j] = -1;
    }

    @Override
    public boolean forbidden(int j) {
        if(G[j] < G[G[j]]) return true;
        for(int i = 0; i < adjMatrix.length; i++){
            if(adjMatrix[j][i] != 0){
                if(G[i] > max[j]) max[j] = G[i];
            }
        }
        if(G[j] < max[j]) {
            return true;
        }
        return false;
    }

    @Override
    public void advance(int j) {
        if(G[j] < G[G[j]]){
            G[j] = G[G[j]];
        }
        else if(G[j] < max[j]){
            G[j] = max[j];
        }
    }

    // This method will be called after solve()
    public int[] getSolution() {
        // Return the vector where the i^th entry is the index j where
        // j is the largest vertex label contained in the component containing 
        // vertex i
        return G;
    }
}
