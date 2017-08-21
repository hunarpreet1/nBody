#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#define BLOCK_SIZE 256
#define SOFTENING 1e-9f
#define w 1000
using namespace cv;

void MyFilledCircle( Mat img, Point center )
{
  circle( img,
      center,
      w/64,
      Scalar( 0, 0, 255 ),
      FILLED,
      LINE_8 );
}

typedef struct { float4 *pos, *vel; } BodySystem;

void randomizeBodies(BodySystem data, int n) {
  for (int i = 0; i < n; i++)
 {
    data.pos[i].x =  w*(rand() / (float)RAND_MAX) ;
    data.pos[i].y = w*(rand() / (float)RAND_MAX);
    data.pos[i].z = w* (rand() / (float)RAND_MAX) ;
    data.vel[i].x = w/20 * (rand() / (float)RAND_MAX) + w/20;
    data.vel[i].y = w /20* (rand() / (float)RAND_MAX) + w/20;
    data.vel[i].z = w/20 * (rand() / (float)RAND_MAX) + w/20;
  }
}

__global__
void bodyForce(float4 *p, float4 *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float3 spos[BLOCK_SIZE];
      float4 tpos = p[tile * blockDim.x + threadIdx.x];
      spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
      __syncthreads();

      for (int j = 0; j < BLOCK_SIZE; j++) {
        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float dz = spos[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += 1000000*dx * invDist3; Fy += 1000000*dy * invDist3; Fz += 1000000*dz * invDist3;
      }
      __syncthreads();
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  int nBodies = 100;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  //const int nIters = 10;  // simulation iterations

  int bytes = 2*nBodies*sizeof(float4);
  float *buf = (float*)malloc(bytes);
  BodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

  randomizeBodies(p, nBodies); // Init pos / vel data
  // p.pos[0].x = w/2;
  // p.pos[0].y = w/4+w/8;
  // p.pos[0].z = 0;
  // p.vel[0].x = w/10;
  // p.vel[0].y = 0;
  // p.vel[0].z = 0;
  // p.pos[1].x = w/2;
  // p.pos[1].y = 3*w/4-w/8;
  // p.pos[1].z = 0;
  // p.vel[1].x = -w/10;
  // p.vel[1].y = 0;
  // p.vel[1].z = 0;

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };
  //randomizeBodies(d_p, nBodies);
  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //double totalTime = 0.0;

  while (true) {

    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies);
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    Mat image = Mat::zeros( w, w, CV_8UC3 );
    namedWindow("nbody",1);

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
      MyFilledCircle(image, Point(p.pos[i].x,p.pos[i].y));
    }

    imshow("nbody",image);
    int c = waitKey(10);
    if(c == 27)
    {
      break;
    }

  }
//     const double tElapsed = GetTimer() / 1000.0;
//     if (iter > 1) { // First iter is warm up
//       totalTime += tElapsed;
//     }
// #ifndef SHMOO
//     printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
// #endif
//   }
//   double avgTime = totalTime / (double)(nIters-1);
//
// #ifdef SHMOO
//   printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
// #else
//   printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per second.\n",
//          nIters, rate);
//   printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
// #endif
  free(buf);
  cudaFree(d_buf);
}
