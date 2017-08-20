#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#define w 1000
#define SOFTENING 1e-9f

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

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(Body *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].x =  w*(rand() / (float)RAND_MAX) ;
    data[i].y = w*(rand() / (float)RAND_MAX);
    data[i].z = w* (rand() / (float)RAND_MAX) ;
    data[i].vx = w/20 * (rand() / (float)RAND_MAX) + w/20;
    data[i].vy = w /20* (rand() / (float)RAND_MAX) + w/20;
    data[i].vz = w/20 * (rand() / (float)RAND_MAX) + w/20;

}
}

void bodyForce(Body *p, float dt, int n) {
  // #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += 10000000*dx * invDist3; Fy += 10000000*dy * invDist3; Fz += 100000*dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  int nBodies = 4096	;
  if (argc > 1) nBodies = atoi(argv[1]);

//   const float dt = 0.01f; // time step
//   const int nIters = 10;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(p, nBodies); // Init pos / vel data
  // p[0].x = w/2;
  // p[0].y = w/4+w/8;
  // p[0].z = 0;
  // p[0].vx = w/10;
  // p[0].vy = 0;
  // p[0].vz = 0;
  // p[1].x = w/2;
  // p[1].y = 3*w/4-w/8;
  // p[1].z = 0;
  // p[1].vx = -w/10;
  // p[1].vy = 0;
  // p[1].vz = 0;
  // p[2].x = 0;
  // p[2].y = w/2;
  // p[2].z = 0;
  // p[2].vx = w/10;
  // p[2].vy = w/10;
  // p[2].vz = 0;
  //double totalTime = 0.0;

  while (true) {
    //StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces

    Mat image = Mat::zeros( w, w, CV_8UC3 );
    namedWindow("nbody",1);


    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
      MyFilledCircle(image,Point(p[i].x,p[i].y));

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
//   //printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per second.\n",
//     //     nIters, rate);
//   printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
// #endif
  free(buf);
  return 0;
}
