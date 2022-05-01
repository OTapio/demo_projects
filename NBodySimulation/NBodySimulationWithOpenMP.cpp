//Copyright by Ossi Tapio 2022
#include <omp.h>
#include <iostream>
#include <random>
#include <cmath>
#include <array>

constexpr int numberOfDimensions = 2;
constexpr float smoothingParameter = 1e-4;
constexpr uint64_t numberOfParticles = 6400*16;
constexpr uint64_t how_many_threads = 32;
constexpr float deltaTime = 0.001;
constexpr float maxSimulationTime = 0.1;

uint64_t particle_i, particle_j, dimension_d;
float acceleration, step;
double start_time, run_time, computeForceTime_A, computeForceTime_B, computePositionTime, computeVelocityTime, computeForce_start_time, computeVelocity_start_time, computePosition_start_time = 0.0;

std::random_device rd;          
std::mt19937 generator(rd()); 
std::uniform_real_distribution<float> distribution_x(0.0,1.0);
std::uniform_real_distribution<float> distribution_v(-1.0,1.0);
std::uniform_real_distribution<float> mass_randomness(1.0,100.0);

struct Particle{

    float mass;
    float position[numberOfDimensions];
    float velocity[numberOfDimensions];
    float Force[numberOfDimensions];
    float Force_old[numberOfDimensions];

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

float sqr(float x) {
    return x*x;
}

void computeForce(std::vector<Particle>& particles) {
    computeForce_start_time = omp_get_wtime();

    omp_set_num_threads(how_many_threads);
    uint64_t sharedNumberOfThreads = omp_get_num_threads();
    #pragma omp parallel private(particle_i, dimension_d) shared(particles, numberOfParticles)
    {
        uint64_t particle_i;
        uint64_t threadIdNumber = omp_get_thread_num();
        float local_sum = 0.0;
        uint64_t localNumberOfThreads = omp_get_num_threads();
        if(threadIdNumber == 0)
        {
           sharedNumberOfThreads=localNumberOfThreads;
        }
        #pragma omp for schedule(dynamic)
        for (particle_i = threadIdNumber; particle_i < numberOfParticles; particle_i += localNumberOfThreads)
        {
            for (dimension_d = 0; dimension_d<numberOfDimensions; dimension_d++) {
                particles[particle_i].Force[dimension_d] = 0;
            }
        }
    }
    double A = omp_get_wtime();
    double B = computeForce_start_time;
    double C = computeForceTime_A;

    computeForceTime_A += omp_get_wtime() - computeForce_start_time;

    float distance_r, auxiliaryResult, completeResult;
#pragma omp parallel private(distance_r , auxiliaryResult, completeResult, particle_i, particle_j, dimension_d) shared(particles, numberOfParticles,smoothingParameter)
    {
        uint64_t particle_i, particle_j;
        uint64_t threadIdNumber = omp_get_thread_num();
        uint64_t localNumberOfThreads = omp_get_num_threads();
        float distance_r, auxiliaryResult, completeResult;
        std::array<float, numberOfDimensions> distances;
        if(threadIdNumber == 0)
        {
            sharedNumberOfThreads = localNumberOfThreads;
        }
        #pragma omp for schedule(dynamic)
        for (particle_i = threadIdNumber; particle_i < numberOfParticles; particle_i += localNumberOfThreads) {
            for (particle_j = particle_i + 1; particle_j < numberOfParticles; particle_j++) {
                distance_r = smoothingParameter;
                for (dimension_d = 0; dimension_d < numberOfDimensions; dimension_d++) {
                    distance_r += sqr(particles[particle_j].position[dimension_d] - particles[particle_i].position[dimension_d]);
                }
                auxiliaryResult = particles[particle_i].mass * particles[particle_j].mass / (std::sqrt(distance_r) * distance_r);
                for (dimension_d = 0; dimension_d < numberOfDimensions; dimension_d++) 
                {
                    completeResult = auxiliaryResult * (particles[particle_j].position[dimension_d] - particles[particle_i].position[dimension_d]);

                    particles[particle_i].Force[dimension_d] += completeResult;
                    particles[particle_j].Force[dimension_d] -= completeResult;
                }
            }
        }
    }
    computeForceTime_B += omp_get_wtime() - computeForce_start_time;
}


void computeVelocity(std::vector<Particle>& particles) 
{    
    computeVelocity_start_time = omp_get_wtime();
    uint64_t sharedNumberOfThreads = omp_get_num_threads();
    #pragma omp parallel private(particle_i, dimension_d, acceleration) shared(particles, numberOfParticles)
    {
        float acceleration;
        uint64_t particle_i, dimension_d;
        uint64_t threadIdNumber = omp_get_thread_num();
        uint64_t localNumberOfThreads = omp_get_num_threads();
        if(threadIdNumber == 0)
        {
            sharedNumberOfThreads = localNumberOfThreads;
        }
        #pragma omp for schedule(dynamic)
        for (particle_i = threadIdNumber; particle_i < numberOfParticles; particle_i += localNumberOfThreads) 
        {
            acceleration = deltaTime * 0.5 / particles[particle_i].mass;
            for (dimension_d = 0; dimension_d<numberOfDimensions; dimension_d++) 
            {
                particles[particle_i].velocity[dimension_d] += acceleration * (particles[particle_i].Force[dimension_d] + particles[particle_i].Force_old[dimension_d]);
            }
        }
    }
    computeVelocityTime += omp_get_wtime() - computeVelocity_start_time;
}

void computePosition(std::vector<Particle>& particles) 
{
    computePosition_start_time = omp_get_wtime();
    uint64_t sharedNumberOfThreads = omp_get_num_threads();
#pragma omp parallel private(particle_i, dimension_d, acceleration) shared(particles, numberOfParticles)
    {
        float acceleration;
        uint64_t particle_i, dimension_d;
        uint64_t id = omp_get_thread_num();
        uint64_t localNumberOfThreads = omp_get_num_threads();
        if(id == 0)
        {
            sharedNumberOfThreads = localNumberOfThreads;
        }
        #pragma omp for schedule(dynamic)
        for (particle_i = 0; particle_i < numberOfParticles; particle_i += localNumberOfThreads) 
        {
            acceleration = deltaTime * 0.5 / particles[particle_i].mass;
            for (dimension_d = 0; dimension_d < numberOfDimensions; dimension_d++) 
            {
                particles[particle_i].position[dimension_d] += deltaTime * (particles[particle_i].velocity[dimension_d] + acceleration * particles[particle_i].Force[dimension_d]);
                particles[particle_i].Force_old[dimension_d] = particles[particle_i].Force[dimension_d];
            }
        }
    }
    computePositionTime += omp_get_wtime() - computePosition_start_time;
}


int main()
{
    float time = 0.0;
    start_time = omp_get_wtime();
    
    std::vector<Particle> particles(numberOfParticles);
    
    uint64_t step;
    std::cout << "number of threads: " << how_many_threads << "  deltaTime: " << deltaTime << "  maxSimulationTime: " << maxSimulationTime << "  numberOfParticles: " << numberOfParticles << "\n";
    computeForce(particles);
    uint64_t numberOfSteps = maxSimulationTime/deltaTime;
    for(step = 0; time < maxSimulationTime; time+=deltaTime, step+=1) {
        computePosition(particles);
        computeForce(particles);
        computeVelocity(particles);

        if (step % 10 == 0) {
            run_time = omp_get_wtime() - start_time;

            std::cout << "\nstep: " << step << "/" << numberOfSteps <<"  current run time: " << run_time << "  computePositionTime: " << computePositionTime << "  computeVelocityTime: " << computeVelocityTime << "  computeForceTime_A: " << computeForceTime_A << "  computeForceTime_B: " << computeForceTime_B << "\n";
        }
    }
    run_time = omp_get_wtime() - start_time;
    std::cout << "total run time: " <<  run_time << "  computePositionTime: " << computePositionTime << "  computeVelocityTime: " << computeVelocityTime << "  computeForceTime_A: " << computeForceTime_A << "  computeForceTime_B: " << computeForceTime_B << "\n";
    
}