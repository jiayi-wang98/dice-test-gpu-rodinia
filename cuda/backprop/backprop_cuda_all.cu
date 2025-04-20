#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>
#include <sys/time.h>

#define BIGRND 0x7fffffff
#define GPU
#define THREADS 256
#define WIDTH 16
#define HEIGHT 16
#define ETA 0.3
#define MOMENTUM 0.3
#define NUM_THREAD 4

typedef struct {
    int input_n;                  /* number of input units */
    int hidden_n;                 /* number of hidden units */
    int output_n;                 /* number of output units */
    float *input_units;          /* the input units */
    float *hidden_units;         /* the hidden units */
    float *output_units;         /* the output units */
    float *hidden_delta;         /* storage for hidden unit error */
    float *output_delta;         /* storage for output unit error */
    float *target;               /* storage for target vector */
    float **input_weights;       /* weights from input to hidden layer */
    float **hidden_weights;      /* weights from hidden to output layer */
    float **input_prev_weights;  /* previous change on input to hidden wgt */
    float **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;

// Function prototypes
void bpnn_initialize(int seed);
BPNN *bpnn_create(int n_in, int n_hidden, int n_out);
void bpnn_free(BPNN *net);
void bpnn_train(BPNN *net, float *eo, float *eh);
void bpnn_feedforward(BPNN *net);
void bpnn_save(BPNN *net, char *filename);
BPNN *bpnn_read(char *filename);
void load(BPNN *net);
void backprop_face();
int setup(int argc, char *argv[]);
void bpnn_train_cuda(BPNN *net, float *eo, float *eh);
float squash(float x);
float **alloc_2d_dbl(int m, int n);
float *alloc_1d_dbl(int n);
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);
void bpnn_randomize_weights(float **w, int m, int n);
void bpnn_randomize_row(float *w, int m);
void bpnn_zero_weights(float **w, int m, int n);
float drnd();
float dpn1();
double gettime();
__global__ void bpnn_layerforward_CUDA(float *input_cuda, float *output_hidden_cuda, float *input_hidden_cuda, float *hidden_partial_sum, int in, int hid);
__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly, int in, float *w, float *oldw);

// Global variables
int layer_size = 0;
unsigned int num_threads = 0;
unsigned int num_blocks = 0;

#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

float drnd() {
    return ((float) rand() / (float) BIGRND);
}

float dpn1() {
    return ((drnd() * 2.0) - 1.0);
}

float squash(float x) {
    return (1.0 / (1.0 + exp(-x)));
}

float *alloc_1d_dbl(int n) {
    float *new_arr = (float *) malloc((unsigned) (n * sizeof(float)));
    if (new_arr == NULL) {
        printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
        return (NULL);
    }
    return (new_arr);
}

float **alloc_2d_dbl(int m, int n) {
    int i;
    float **new_arr = (float **) malloc((unsigned) (m * sizeof(float *)));
    if (new_arr == NULL) {
        printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
        return (NULL);
    }
    for (i = 0; i < m; i++) {
        new_arr[i] = alloc_1d_dbl(n);
    }
    return (new_arr);
}

void bpnn_randomize_weights(float **w, int m, int n) {
    int i, j;
    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i][j] = (float) rand() / RAND_MAX;
        }
    }
}

void bpnn_randomize_row(float *w, int m) {
    int i;
    for (i = 0; i <= m; i++) {
        w[i] = 0.1;
    }
}

void bpnn_zero_weights(float **w, int m, int n) {
    int i, j;
    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i][j] = 0.0;
        }
    }
}

void bpnn_initialize(int seed) {
    printf("Random number generator seed: %d\n", seed);
    srand(seed);
}

BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out) {
    BPNN *newnet = (BPNN *) malloc(sizeof(BPNN));
    if (newnet == NULL) {
        printf("BPNN_CREATE: Couldn't allocate neural network\n");
        return (NULL);
    }
    newnet->input_n = n_in;
    newnet->hidden_n = n_hidden;
    newnet->output_n = n_out;
    newnet->input_units = alloc_1d_dbl(n_in + 1);
    newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
    newnet->output_units = alloc_1d_dbl(n_out + 1);
    newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
    newnet->output_delta = alloc_1d_dbl(n_out + 1);
    newnet->target = alloc_1d_dbl(n_out + 1);
    newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
    newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
    return (newnet);
}

void bpnn_free(BPNN *net) {
    int n1 = net->input_n;
    int n2 = net->hidden_n;
    int i;
    free((char *) net->input_units);
    free((char *) net->hidden_units);
    free((char *) net->output_units);
    free((char *) net->hidden_delta);
    free((char *) net->output_delta);
    free((char *) net->target);
    for (i = 0; i <= n1; i++) {
        free((char *) net->input_weights[i]);
        free((char *) net->input_prev_weights[i]);
    }
    free((char *) net->input_weights);
    free((char *) net->input_prev_weights);
    for (i = 0; i <= n2; i++) {
        free((char *) net->hidden_weights[i]);
        free((char *) net->hidden_prev_weights[i]);
    }
    free((char *) net->hidden_weights);
    free((char *) net->hidden_prev_weights);
    free((char *) net);
}

BPNN *bpnn_create(int n_in, int n_hidden, int n_out) {
    BPNN *newnet = bpnn_internal_create(n_in, n_hidden, n_out);
    bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
    bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
    bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
    bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
    bpnn_randomize_row(newnet->target, n_out);
    return (newnet);
}

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2) {
    float sum;
    int j, k;
    l1[0] = 1.0;
    for (j = 1; j <= n2; j++) {
        sum = 0.0;
        for (k = 0; k <= n1; k++) {
            sum += conn[k][j] * l1[k];
        }
        l2[j] = squash(sum);
    }
}

void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err) {
    int j;
    float o, t, errsum = 0.0;
    for (j = 1; j <= nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += ABS(delta[j]);
    }
    *err = errsum;
}

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err) {
    int j, k;
    float h, sum, errsum = 0.0;
    for (j = 1; j <= nh; j++) {
        h = hidden[j];
        sum = 0.0;
        for (k = 1; k <= no; k++) {
            sum += delta_o[k] * who[j][k];
        }
        delta_h[j] = h * (1.0 - h) * sum;
        errsum += ABS(delta_h[j]);
    }
    *err = errsum;
}

void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw) {
    float new_dw;
    int k, j;
    ly[0] = 1.0;
    for (j = 1; j <= ndelta; j++) {
        for (k = 0; k <= nly; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
            w[k][j] += new_dw;
            oldw[k][j] = new_dw;
        }
    }
}

void bpnn_feedforward(BPNN *net) {
    int in = net->input_n;
    int hid = net->hidden_n;
    int out = net->output_n;
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
}

void bpnn_train(BPNN *net, float *eo, float *eh) {
    int in = net->input_n;
    int hid = net->hidden_n;
    int out = net->output_n;
    float out_err, hid_err;
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
    *eo = out_err;
    *eh = hid_err;
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
}

void bpnn_save(BPNN *net, char *filename) {
    int n1 = net->input_n, n2 = net->hidden_n, n3 = net->output_n, i, j, memcnt;
    float dvalue, **w;
    char *mem;
    FILE *pFile = fopen(filename, "w+");
    printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
    fwrite((char *) &n1, sizeof(char), sizeof(char), pFile);
    fwrite((char *) &n2, sizeof(char), sizeof(char), pFile);
    fwrite((char *) &n3, sizeof(char), sizeof(char), pFile);
    memcnt = 0;
    w = net->input_weights;
    mem = (char *) malloc((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            dvalue = w[i][j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(float));
            memcnt += sizeof(float);
        }
    }
    fwrite(mem, sizeof(float), (unsigned) ((n1+1) * (n2+1)), pFile);
    free(mem);
    memcnt = 0;
    w = net->hidden_weights;
    mem = (char *) malloc((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            dvalue = w[i][j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(float));
            memcnt += sizeof(float);
        }
    }
    fwrite(mem, sizeof(float), (unsigned) ((n2+1) * (n3+1)), pFile);
    free(mem);
    fclose(pFile);
}

BPNN *bpnn_read(char *filename) {
    char *mem;
    BPNN *new_net;
    int fd, n1, n2, n3, i, j, memcnt;
    if ((fd = open(filename, 0, 0644)) == -1) {
        return (NULL);
    }
    printf("Reading '%s'\n", filename);
    read(fd, (char *) &n1, sizeof(int));
    read(fd, (char *) &n2, sizeof(int));
    read(fd, (char *) &n3, sizeof(int));
    new_net = bpnn_internal_create(n1, n2, n3);
    printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
    printf("Reading input weights...");
    memcnt = 0;
    mem = (char *) malloc((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
    read(fd, mem, (n1+1) * (n2+1) * sizeof(float));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            fastcopy(&(new_net->input_weights[i][j]), &mem[memcnt], sizeof(float));
            memcnt += sizeof(float);
        }
    }
    free(mem);
    printf("Done\nReading hidden weights...");
    memcnt = 0;
    mem = (char *) malloc((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
    read(fd, mem, (n2+1) * (n3+1) * sizeof(float));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            fastcopy(&(new_net->hidden_weights[i][j]), &mem[memcnt], sizeof(float));
            memcnt += sizeof(float);
        }
    }
    free(mem);
    close(fd);
    printf("Done\n");
    bpnn_zero_weights(new_net->input_prev_weights, n1, n2);
    bpnn_zero_weights(new_net->hidden_prev_weights, n2, n3);
    return (new_net);
}

void load(BPNN *net) {
    float *units;
    int nr, imgsize, i, k;
    nr = layer_size;
    imgsize = nr;
    units = net->input_units;
    k = 1;
    for (i = 0; i < nr; i++) {
        units[k] = (float) rand() / RAND_MAX;
        k++;
    }
}

void backprop_face() {
    BPNN *net;
    float out_err, hid_err;
    net = bpnn_create(layer_size, 16, 1);
    printf("Input layer size : %d\n", layer_size);
    load(net);
    printf("Starting training kernel\n");
    bpnn_train_cuda(net, &out_err, &hid_err);
    bpnn_free(net);
    printf("Training done\n");
}

int setup(int argc, char *argv[])
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

__global__ void bpnn_layerforward_CUDA(float *input_cuda, float *output_hidden_cuda, float *input_hidden_cuda, float *hidden_partial_sum, int in, int hid) {
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_in = HEIGHT * by + ty + 1;
    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];
    if (tx == 0)
        input_node[ty] = input_cuda[index_in];
    __syncthreads();
    weight_matrix[ty][tx] = input_hidden_cuda[index];
    __syncthreads();
    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
    __syncthreads();
    for (int i = 1; i <= __log2f(HEIGHT); i++) {
        int power_two = __powf(2, i);
        if (ty % power_two == 0)
            weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
        __syncthreads();
    }
    input_hidden_cuda[index] = weight_matrix[ty][tx];
    __syncthreads();
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
    }
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly, int in, float *w, float *oldw) {
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;
    w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
    oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
    __syncthreads();
    if (ty == 0 && by == 0) {
        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    }
}

void bpnn_train_cuda(BPNN *net, float *eo, float *eh) {
    int in = net->input_n;
    int hid = net->hidden_n;
    int out = net->output_n;
    float out_err, hid_err;
    int m = 0;
    float *input_hidden_cuda;
    float *input_cuda;
    float *output_hidden_cuda;
    float *partial_sum;
    float *hidden_partial_sum;
    float *hidden_delta_cuda;
    float *input_prev_weights_cuda;
    float sum;
    float *input_weights_one_dim;
    float *input_weights_prev_one_dim;
    num_blocks = in / 16;
    dim3 grid(1, num_blocks);
    dim3 threads(16, 16);
    input_weights_one_dim = (float *) malloc((in + 1) * (hid + 1) * sizeof(float));
    input_weights_prev_one_dim = (float *) malloc((in + 1) * (hid + 1) * sizeof(float));
    partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
    for (int k = 0; k <= in; k++) {
        for (int j = 0; j <= hid; j++) {
            input_weights_one_dim[m] = net->input_weights[k][j];
            input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
            m++;
        }
    }
    cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
    cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
    cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
    cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
    printf("Performing GPU computation\n");
    cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda, output_hidden_cuda, input_hidden_cuda, hidden_partial_sum, in, hid);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
    for (int j = 1; j <= hid; j++) {
        sum = 0.0;
        for (int k = 0; k < num_blocks; k++) {
            sum += partial_sum[k * hid + j-1];
        }
        sum += net->input_weights[0][j];
        net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
    cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));
    cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda, hid, input_cuda, in, input_hidden_cuda, input_prev_weights_cuda);
    cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_cuda);
    cudaFree(output_hidden_cuda);
    cudaFree(input_hidden_cuda);
    cudaFree(hidden_partial_sum);
    cudaFree(input_prev_weights_cuda);
    cudaFree(hidden_delta_cuda);
    free(partial_sum);
    free(input_weights_one_dim);
    free(input_weights_prev_one_dim);
    *eo = out_err;
    *eh = hid_err;
}

int main(int argc, char** argv) {
    setup(argc, argv);
    return 0;
}