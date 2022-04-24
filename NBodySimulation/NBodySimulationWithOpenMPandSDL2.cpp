#include "SDL2/SDL.h"
#include "SDL2/SDL_video.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream> 
#include <random>
#include <omp.h>

#define fps 60
#define window_width 640*2
#define window_height 480*2
#define pi 3.1415926535
#define NUM_THREADS 32
//#include "nbody.h"


// Parameters
const int numberOfDimensions = 2;          // dimensions
const double smoothingParameter = 1e-4;    // smoothing parameter
const uint64_t numberOfParticles = 160;
uint64_t how_many_threads = NUM_THREADS;
// Function prototypes
inline double sqr(double);



// Other Functions

inline double sqr(double x) {
    return x*x;
}

std::random_device rd;          
std::mt19937 generator(rd()); 
std::uniform_real_distribution<double> distribution_x(0.0,window_width);
std::uniform_real_distribution<double> distribution_y(0.0,window_height);
std::uniform_real_distribution<double> distribution_v(-1.0,1.0);
std::uniform_real_distribution<double> mass_randomness(1.0,1000.0);

struct Particle{

    double mass;           // mass
    double position[numberOfDimensions];      // position
    double velocity[numberOfDimensions];      // velocity
    double Force[numberOfDimensions];      // force 
    double Force_old[numberOfDimensions];  // force past time step

    Particle()
    {
        mass = (1./numberOfParticles) * mass_randomness(generator);
        position[0] = distribution_x(generator);
        position[1] = distribution_y(generator);
        for (uint64_t dimension_d=0; dimension_d<numberOfDimensions; dimension_d++)
        {
            
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
    #pragma omp parallel// for ordered schedule(static)
    {
        uint64_t particle_i;
        uint64_t id=omp_get_thread_num();
        double local_sum=0.0;
        uint64_t local_num_threads=omp_get_num_threads(); //"nthrds" in video

        if(id==0)//only do once
        shared_num_threads=local_num_threads;
        #pragma omp parallel for private(particle_i, dimension_d)  shared(particles, numberOfParticles)//ordered schedule(static)
        for (particle_i=id; particle_i < numberOfParticles; particle_i += local_num_threads)
        {
            for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) {
                particles[particle_i].Force[dimension_d] = 0;
            }
        }
    }

    std::vector<Particle> particlesResults(numberOfParticles);
    double r,f;

#pragma omp parallel for private(r,f,particle_i, particle_j, dimension_d)  shared(particles,particlesResults, numberOfParticles,smoothingParameter)//ordered schedule(static)
    for (particle_i=0; particle_i<numberOfParticles; particle_i++) {
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


void comp_velocity(std::vector<Particle>& particles, const double deltaTime) 
{    
    uint64_t particle_i, dimension_d;
    double acceleration;
    #pragma omp parallel for private(acceleration ,particle_i, dimension_d)  shared(particles, numberOfParticles, deltaTime)//ordered schedule(static)
    for (particle_i=0; particle_i<numberOfParticles; particle_i++) 
    {
        acceleration = deltaTime * 0.5 / particles[particle_i].mass;
        for (dimension_d=0; dimension_d<numberOfDimensions; dimension_d++) 
        {
            particles[particle_i].velocity[dimension_d] += acceleration * (particles[particle_i].Force[dimension_d] + particles[particle_i].Force_old[dimension_d]);
        }
    }
}

void comp_position(std::vector<Particle>& particles, const double deltaTime) 
{
    uint64_t particle_i, dimension_d;
    double acceleration;
    #pragma omp parallel for private(acceleration ,particle_i, dimension_d)  shared(particles, numberOfParticles, deltaTime)//ordered schedule(static)
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

/*int main() {
    //Nbody nbody(16, 0.001, 0.1);
    //nbody.timeIntegration();

}*/

void cap_framerate(uint32_t starting_tick)
{
    if ((1000 / fps)>SDL_GetTicks() - starting_tick){
        SDL_Delay(1000/fps - (SDL_GetTicks() - starting_tick));
    }
}

class Sprite
{

    protected:

        SDL_Surface *image;
        SDL_Rect rect;

        int origin_x, origin_y;


    public:

        Sprite(Uint32 color, int x, int y, int w = 10, int h = 10)
        {
            image = SDL_CreateRGBSurface(0, w, h, 32, 0, 0, 0, 0);

            SDL_FillRect( image, NULL, color);

            rect = image->clip_rect;

            int origin_x = rect.w/2;
            int origin_y = rect.h/2;

            rect.x = x;// - origin_x;
            rect.y = y;// - origin_y;
        }
        
        ~Sprite()
        {

        }

        void update(){
            // Can be overridden!
        }

        void draw(SDL_Surface *destination){
            SDL_BlitSurface(image,NULL, destination, &rect);

        }

        SDL_Surface* get_image() const
        {
            return image;
        }

        bool operator==(const Sprite &other) const
        {
            return (image == other.get_image());
        }

};

class SpriteGroup
{

    private:

    std::vector<Sprite*> sprites;

    int sprites_size; 

    public:

    SpriteGroup copy()
    {
        SpriteGroup new_group;

        for (int i = 0; i < sprites_size; i++)
        {
            new_group.add(sprites[i]);
        }
        return new_group;
    }

    void add(Sprite *sprite)
    {

            sprites.push_back(sprite);

            sprites_size = sprites.size();

    }

    void remove(Sprite sprite_object)
    {
        for (int i = 0; i < sprites_size; i++)
        {
            if (*sprites[i] == sprite_object)
            {
                sprites.erase( sprites.begin() + i);
            }
        }

        sprites_size = sprites.size();
    }


    bool has(Sprite sprite_object)
    {
        for (int i = 0; i < sprites_size; i++)
        {
            if (*sprites[i] == sprite_object)
            {
                return true;
            }
        }
        return false;
    }

    void update()
    {
        if(! sprites.empty())
        {
            for (int i = 0; i < sprites_size; i++)
            {

                sprites[i]->update();

            }
        }
    }

    void draw(SDL_Surface *destination)
    {
        if(! sprites.empty())
        {
            for (int i = 0; i < sprites_size; i++)
            {
                sprites[i]->draw(destination);
            }
        }
    }

    void empty()
    {   
        sprites.clear();
        sprites_size = sprites.size();
    }

    int size()
    {
        return sprites_size;
    }

    std::vector<Sprite*> get_sprites()
    {
        return sprites;
    }


};


class Block : public Sprite
{

    public:
        Block(Uint32 color, int x, int y, int w = 48, int h = 64) : Sprite(color, x, y, w, h)
        
        {
            update_properties();
        }
        ~Block()
        {

        }

        void update_properties()
        {
            origin_x = 0;
            origin_y = 0;

            set_position(rect.x, rect.y);
        }

        void set_position(int x, int y)
        {
            rect.x = x - origin_x;
            rect.y = y - origin_y;

        }

        void set_image(const char filename[] = NULL)
        {
            if (filename != NULL){
                SDL_Surface *loaded_image = NULL;

                loaded_image = SDL_LoadBMP(filename);

                if (loaded_image != NULL) {
                    
                    image = loaded_image;

                    int old_x = rect.x;
                    int old_y = rect.y;

                    rect = image->clip_rect;

                    rect.x = old_x;
                    rect.y = old_y;

                    update_properties();
                }
            }
        }

    private:

};

class Kappale
{

    public:
        Kappale(const float m, const float koord_x, const float koord_y , const float v_x, const float v_y)
        {
            massa = m;
            koordinaatti_x = koord_x;
            koordinaatti_y = koord_y;

            nopeus_x = v_x;
            nopeus_y = v_y;
            paivita_nopeus(v_x,v_y);
            paivita_liikemaara();
            paivita_kineettinen_energia();
        }

        ~Kappale()
        {

        }

        void paivita_nopeus(const float v_x, const float v_y)
        {
            kok_nopeus = std::sqrt(v_x*v_x + v_y*v_y);
        }
        
        void paivita_liikemaara()
        {
            liikemaara_x = nopeus_x * massa;
            liikemaara_y = nopeus_y * massa;
            kok_liikemaara = liikemaara_x + liikemaara_y;
        }

        void paivita_sijainti(const float Dt)
        {
            koordinaatti_x = koordinaatti_x + nopeus_x * Dt;
            koordinaatti_y = koordinaatti_y + nopeus_y * Dt;
        }

        void paivita_kineettinen_energia()
        {
            kineettinen_energia_x = liikemaara_x * liikemaara_x / (2*massa);
            kineettinen_energia_y = liikemaara_y * liikemaara_y / (2*massa);
            kok_kineettinen_energia = kineettinen_energia_x + kineettinen_energia_y;
        }

    //private:


        float koordinaatti_x;
        float koordinaatti_y;

        float massa;

        float liikemaara_x;
        float liikemaara_y;
        float kok_liikemaara;

        float nopeus_x;
        float nopeus_y;
        float kok_nopeus;

        float kineettinen_energia_x;
        float kineettinen_energia_y;
        float kok_kineettinen_energia;

    private:
};

int main( int argc, char* args[] )
{
    //Nbody nbody(1600, 0.001, 0.1);
    std::vector<Particle> particles(numberOfParticles);
    const double deltaTime = 0.001;
    const double maxSimulationTime = 0.1;
    uint64_t step = 0;
    double time = 0;

    SDL_Init( SDL_INIT_EVERYTHING );
    
    SDL_Window *window = NULL;
    SDL_Renderer* renderer = NULL;

    window = SDL_CreateWindow(
        "Minun SDL-ikkuna",                  // window title
        SDL_WINDOWPOS_UNDEFINED,           // initial x position        //nbody.timeIntegration();
    );

    if (window == NULL) {
        std::cout << "Could not create window: %s\n" << SDL_GetError();
        return 1;
    }

    SDL_Surface *screen = SDL_GetWindowSurface(window);

    Uint32 black    = SDL_MapRGB(screen->format,  0,  0,  0);
    Uint32 almost_black    = SDL_MapRGB(screen->format,  10,  10,  10);
    Uint32 white    = SDL_MapRGB(screen->format,255,255,255);
    Uint32 greyish  = SDL_MapRGB(screen->format,200,200,200);
    Uint32 red      = SDL_MapRGB(screen->format,255,  0,  0);
    Uint32 red2      = SDL_MapRGB(screen->format,255,127,127);
    Uint32 green    = SDL_MapRGB(screen->format,  0,255,  0);
    Uint32 green2    = SDL_MapRGB(screen->format,127,255,127);
    Uint32 blue     = SDL_MapRGB(screen->format,  0,  0,255);
    Uint32 blue2     = SDL_MapRGB(screen->format,127,127,255);
    Uint32 cyan     = SDL_MapRGB(screen->format,  0,255,255);
    Uint32 cyan2     = SDL_MapRGB(screen->format,127,255,255);
    Uint32 magenta  = SDL_MapRGB(screen->format,255,  0,255);
    Uint32 magenta2  = SDL_MapRGB(screen->format,255,127,255);
    Uint32 yellow   = SDL_MapRGB(screen->format,255,255,  0);
    Uint32 yellow2   = SDL_MapRGB(screen->format,255,255,127);

    SDL_FillRect(screen, NULL, almost_black);
    
    std::vector<Block> hiukkaset;
    for(uint32_t i = 0; i < numberOfParticles; i++)
    {
        hiukkaset.emplace_back(white, particles[i].position[0], particles[i].position[1], 5 * (particles[i].mass), 5 * (particles[i].mass));
    }
    

    SpriteGroup active_sprites;

    for(uint32_t i = 0; i < numberOfParticles; i++)
    {
        active_sprites.add(&hiukkaset[i]);
    }

    active_sprites.draw(screen); 

    SDL_UpdateWindowSurface(window);



    uint32_t starting_tick;
    SDL_Event test_event;
    bool event_running = true;    

    uint64_t aika, laskuri = 0;
    comp_force(particles);
    while(event_running)
    {
        starting_tick = SDL_GetTicks();
        while (SDL_PollEvent(&test_event))
        {
            if(test_event.type == SDL_QUIT )
            {
                event_running = false;
                break;
            }
        }
        
        laskuri++;
        aika++;

        comp_position(particles, deltaTime);
        comp_force(particles);
        comp_velocity(particles, deltaTime);
        if (laskuri == 600)
        {
            std::cout << "\n";
            std::cout << "particles[0].position[0]: " << particles[0].position[0] << " particles[0].position[1]: " <<  particles[0].position[1] << "\n";
            std::cout << "deltaTime: " << deltaTime << "\n";
            std::cout << "maxSimulationTime: " << maxSimulationTime << "\n";
            std::cout << "numberOfParticles: " << numberOfParticles << "\n";
            std::cout << "step: " << step << "\n";
            std::cout << "particles[0].Force[0]: " << particles[0].Force[0] << "\n";
            std::cout << "particles[0].Force_old[0]:" << particles[0].Force_old[0] << "\n";
            std::cout << "particles[0].mass: " << particles[0].mass << "\n";
            std::cout << "particles[0].velocity[0]: " << particles[0].velocity[0] << "\n\n";
            laskuri = 0;
            
        }

        SDL_FillRect(screen, NULL, almost_black);
   
        for(uint32_t i = 0; i < numberOfParticles; i++)
        {
            hiukkaset[i].set_position(particles[i].position[0], particles[i].position[1]);
        }
        
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        SDL_RenderDrawLine(renderer, 320, 200, 300, 240);
        SDL_RenderDrawLine(renderer, 300, 240, 340, 240);
        SDL_RenderDrawLine(renderer, 340, 240, 320, 200);
        SDL_RenderPresent(renderer);

        active_sprites.draw(screen);
        SDL_UpdateWindowSurface(window);
    }


    SDL_DestroyRenderer(renderer);
    // Close and destroy the window
    SDL_DestroyWindow(window);

    //Quit SDL
    SDL_Quit();
    
    return 0;    
}

