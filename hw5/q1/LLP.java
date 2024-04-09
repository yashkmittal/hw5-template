import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public abstract class LLP {
    // Feel free to add any methods here. Common parameters (e.g. number of processes)
    // can be passed up through a super constructor. Your code will be tested by creating
    // an instance of a sub-class, calling the solve() method below, and then calling the
    // sub-class's getSolution() method. You are free to modify anything else as long as
    // you follow this API (see SimpleTest.java)

    // This is the same as the size of the vector processed (i.e, each thread will process a different index of the vector G)
    public final int numThreads;
    public int G[];

    public LLP(int size) {
        this.numThreads = size;
        this.G = new int[size];
    }

    // Checks whether process j is forbidden in the state vector G
    public abstract boolean forbidden(int j);

    // Advances on process j
    public abstract void advance(int j);

    //Initializes G with a value specified in the sub-class
    public abstract void init(int j);

    public void solve() {
        // Implement this method. There are many ways to do this but you
        // should follow the following basic steps:
        // 1. Compute the forbidden states
        // 2. Advance on forbidden states in parallel
        // 3. Repeat 1 and 2 until there are no forbidden states
        
        // This method will be called before getSolution()
        
        AtomicBoolean forbidden = new AtomicBoolean(true);
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for(int i = 0; i < numThreads; i++){
            final int j = i;
            executor.submit(new Runnable() {
                public void run() {
                    init(j);
                }
            });
        }
        executor.shutdown();
        // for(int i = 0; i < numThreads; i++){
        //     init(i);
        // }
        
        
        while (forbidden.get()) {
            executor = Executors.newFixedThreadPool(numThreads);
            forbidden.set(false);
            for (int i = 0; i < numThreads; i++) {
                // This is parallel version using the ExecutorService
                final int j = i;
                executor.submit(new Runnable() {
                    public void run() {
                        if (forbidden(j)) {
                            forbidden.set(true);
                            advance(j);
                        }
                    }
                });
                // This is a sequential version of the above code, for debugging
                // if(forbidden(i)){
                //     forbidden.set(true);
                //     advance(i);
                // }
            }
            executor.shutdown();
            try {
                executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        forbidden.set(false);
    }
}
