//Copyright by Ossi Tapio 2022
#include <omp.h>
#include <iostream>
#include <random>

double step;


constexpr int numberOfDimensions = 2;          // dimensions
constexpr double smoothingParameter = 1e-4;    // smoothing parameter
constexpr uint64_t numberOfParticles = 6400;
constexpr uint64_t how_many_threads = 32;
constexpr double deltaTime = 0.001;
constexpr double maxSimulationTime = 0.1;
// Function prototypes
inline double sqr(double);

std::random_device rd;          
std::mt19937 generator(rd()); 
std::uniform_real_distribution<double> distribution_x(0.0,1.0);
std::uniform_real_distribution<double> distribution_v(-1.0,1.0);
std::uniform_real_distribution<double> mass_randomness(1.0,100.0);

struct Particle{

    double mass;           // mass
    double position[numberOfDimensions];      // position
    double velocity[numberOfDimensions];      // velocity
    double Force[numberOfDimensions];      // force 
    double Force_old[numberOfDimensions];  // force past time step

    Particle()
    {
        mass = (1./numberOfParticles) * mass_randomness(generator);
        for (uint64_t dimension_d=0; dimension_d<numberOfDimensions; dimension_d++)
        {
            position[dimension_d] = distribution_x(generator);
            velocity[dimension_d] = distribution_v(generator);
            Force[dimension_d] = 0.0;
            Force_old[dimension_d] = 0.0;
        }
    }

    ~Particle(){}
};


void comp_force(std::vector<Particle>& particles) {
    uint64_t particle_i, particle_j;
    uint64_t dimension_d;
    omp_set_num_threads(how_many_threads);
    uint64_t shared_num_threads=omp_get_num_threads(); //"nthreads" in video
    #pragma omp parallel private(particle_i, dimension_d) shared(particles, numberOfParticles)// for ordered schedule(static)
    {
        uint64_t particle_i;
        uint64_t id=omp_get_thread_num();
        double local_sum=0.0;
        uint64_t local_num_threads=omp_get_num_threads(); //"nthrds" in video
        if(id==0)//only do once
        {
           shared_num_threads=local_num_threads;
        }
        #pragma omp for
        for (particle_i=id; particle_i < numberOfParticles; particle_i += local_num_threads)
        {
            for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) {
                particles[particle_i].Force[dimension_d] = 0;
            }
        }
    }

    double r,f;
#pragma omp parallel private(r,f,particle_i, particle_j, dimension_d) shared(particles, numberOfParticles,smoothingParameter)// ordered schedule(static)
{
    uint64_t particle_i, particle_j;
    uint64_t id=omp_get_thread_num();
    double local_sum=0.0;
    uint64_t local_num_threads=omp_get_num_threads(); //"nthrds" in video
    double r,f;
    if(id==0)//only do once
    {
        shared_num_threads=local_num_threads;
    }
    #pragma omp for
    for (particle_i=id; particle_i < numberOfParticles; particle_i += local_num_threads) {
        for (particle_j=particle_i+1; particle_j<numberOfParticles; particle_j++) {
            r=smoothingParameter; // smoothing
            for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) {
                r += sqr(particles[particle_j].position[dimension_d] - particles[particle_i].position[dimension_d]);
            }
            f = particles[particle_i].mass * particles[particle_j].mass / (sqrt(r) * r);
            for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) 
            {
                particles[particle_i].Force[dimension_d] += f * (particles[particle_j].position[dimension_d] - particles[particle_i].position[dimension_d]);
                particles[particle_j].Force[dimension_d] -= f * (particles[particle_j].position[dimension_d] - particles[particle_i].position[dimension_d]);
            }
        }
    }
}
}





void comp_velocity(std::vector<Particle>& particles, const double deltaTime) 
{    
    uint64_t particle_i, dimension_d;
    double acceleration;
    uint64_t shared_num_threads=omp_get_num_threads();
    //#pragma omp parallel for private(acceleration ,particle_i, dimension_d)  shared(particles, numberOfParticles, deltaTime)//ordered schedule(static)
#pragma omp parallel private(particle_i, dimension_d, acceleration) shared(particles, numberOfParticles)// ordered schedule(static)
{
    double acceleration;
    uint64_t particle_i, dimension_d;
    uint64_t id=omp_get_thread_num();
    double local_sum=0.0;
    uint64_t local_num_threads=omp_get_num_threads(); //"nthrds" in video
    if(id==0)//only do once
    {
        shared_num_threads=local_num_threads;
    }
    #pragma omp for
    for (particle_i=id; particle_i<numberOfParticles; particle_i += local_num_threads) 
    {
        acceleration = deltaTime * 0.5 / particles[particle_i].mass;
        for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) 
        {
            particles[particle_i].velocity[dimension_d] += acceleration * (particles[particle_i].Force[dimension_d] + particles[particle_i].Force_old[dimension_d]);
        }
    }
}
}

void comp_position(std::vector<Particle>& particles, const double deltaTime) 
{
    uint64_t particle_i, dimension_d;
    double acceleration;
    uint64_t shared_num_threads=omp_get_num_threads();
    //#pragma omp parallel for private(acceleration ,particle_i, dimension_d)  shared(particles, numberOfParticles, deltaTime)//ordered schedule(static)
#pragma omp parallel private(particle_i, dimension_d, acceleration) shared(particles, numberOfParticles)// ordered schedule(static)
{
    double acceleration;
    uint64_t particle_i, dimension_d;
    uint64_t id=omp_get_thread_num();
    double local_sum=0.0;
    uint64_t local_num_threads=omp_get_num_threads(); //"nthrds" in video
    if(id==0)//only do once
    {
        shared_num_threads=local_num_threads;
    }
    #pragma omp for
    for (particle_i=0; particle_i<numberOfParticles; particle_i++) 
    {
        acceleration = deltaTime * 0.5 / particles[particle_i].mass;
        for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) 
        {
            particles[particle_i].position[dimension_d] += deltaTime * (particles[particle_i].velocity[dimension_d] + acceleration * particles[particle_i].Force[dimension_d]);
            particles[particle_i].Force_old[dimension_d] = particles[particle_i].Force[dimension_d];
        }
    }
}
}


// Other Functions

inline double sqr(double x) {
    return x*x;
}

int main()
{
    double start_time = omp_get_wtime();
    
    std::vector<Particle> particles(numberOfParticles);
    uint64_t step = 0;
    double time = 0;
    
    std::cout << "number of threads: " << how_many_threads << "  deltaTime: " << deltaTime << "  maxSimulationTime: " << maxSimulationTime << "  numberOfParticles: " << numberOfParticles << "\n";
    comp_force(particles);
    uint64_t numberOfSteps = maxSimulationTime/deltaTime;
    for(; time<maxSimulationTime; time+=deltaTime, step+=1) {
        comp_position(particles, deltaTime);
        comp_force(particles);
        comp_velocity(particles, deltaTime);

        if (step % 10 == 0) {
            std::cout << "\nB step: " << step << "/" << numberOfSteps <<"\n";
        }
    }
    double run_time = omp_get_wtime() - start_time;
    std::cout << "total runtime: " << run_time << "\n";
}