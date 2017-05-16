/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // 1 
    num_particles = 150;

    particles.resize(num_particles);
    weights.resize(num_particles, 0.0);
    random_device rnd;
    default_random_engine rng(rnd());    // random number generator
    rng.seed(123);

    // 2 - normal dist for x
    normal_distribution<double>     x_dist(x, std[0]);
    normal_distribution<double>     y_dist(y, std[1]);
    normal_distribution<double>     theta_dist(theta, std[2]);

    for (int k = 0; k < num_particles; k++) {
        Particle    particle;

        particle.id     = k;
        particle.x      = x_dist(rng);
        particle.y      = y_dist(rng);
        particle.theta  = theta_dist(rng);
        particle.weight = 1.0 / num_particles;

        particles[k]     = particle;         // particles.push_back(particle);
        weights[k]       = particle.weight;     // .push_back(particle.weight);
    }

    // DEBUG
    // std::cout << "Num particles: " << particles.size() << endl;
    // std::cout << "Num weights: " << weights.size() << endl;

    is_initialized = true;
    return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    random_device rnd;
    default_random_engine rng(rnd());    // random number generator
    rng.seed(456);

    const double velo_dt    = velocity * delta_t;
    const double yaw_dt     = yaw_rate * delta_t;
    const double vel_yaw    = velocity / yaw_rate;

    // for (int k = 0; k < num_particles; k++) {
    for (auto& p: particles) {

        // Particle p        = particles[k];

        if (fabs(yaw_rate) > 1e-4) {
            const double theta_new = p.theta + yaw_dt;
            p.x     += vel_yaw * (sin(theta_new) - sin(p.theta));
            p.y     += vel_yaw * (cos(p.theta) - cos(theta_new));
            p.theta = theta_new;
        } else {
            p.x     += velo_dt * cos(p.theta);
            p.y     += velo_dt * sin(p.theta);
        }

        
        normal_distribution<double>    x_dist(p.x, std_pos[0]);
        normal_distribution<double>    y_dist(p.y, std_pos[1]);
        normal_distribution<double>    theta_dist(p.theta, std_pos[2]);

        p.x     = x_dist(rng);
        p.y     = y_dist(rng);
        p.theta = theta_dist(rng);

    } // end-for-particles
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
}

std::vector<LandmarkObs> associateLandmarks(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    vector<LandmarkObs> nearest_landmarks;
    LandmarkObs         closestLM;

    for (auto anObservation: observations) {
        double    shortest = 1E12;
        // find distance from an observation to a prediction; if this distance is the shortest, 
        // save this landmark in the nearest_landmarks list
        for (auto aPrediction: predicted) {
            double current_distance = dist(anObservation.x, anObservation.y, aPrediction.x, aPrediction.y);
            if (current_distance < shortest) {
                shortest     = current_distance;
                closestLM    = aPrediction;
            }
        }

        nearest_landmarks.push_back(closestLM);
    }

    return nearest_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html


    double    std_x = std_landmark[0];
    double    std_y = std_landmark[1];

    // optimization
    const double std_x_sq    = 2 * std_x * std_x;
    const double std_y_sq    = 2 * std_y * std_y;
    const double std_xy_2PI  = 2 * M_PI * std_x * std_y;
    
    for (int k = 0; k < particles.size(); k++) {

        Particle a_particle     = particles[k];

        // 1 - transform observations from Vehicle coords to Map coords
        vector<LandmarkObs> transformed_observations;
        // 1a - for each observation, transform it
        for (auto anObservation: observations) {
            LandmarkObs tx_observation;    
            tx_observation.x     = a_particle.x + anObservation.x * cos(a_particle.theta) - anObservation.y * sin(a_particle.theta);
            tx_observation.y     = a_particle.y + anObservation.x * sin(a_particle.theta) + anObservation.y * cos(a_particle.theta);
            tx_observation.id    = anObservation.id;

            transformed_observations.push_back(tx_observation);
        }

        // 2 - find landmarks closest to the particle
        vector<LandmarkObs> predicted_landmarks;
        for (auto aLandmark: map_landmarks.landmark_list) {

            double current_distance = dist(a_particle.x, a_particle.y, aLandmark.x_f, aLandmark.y_f);
            if (current_distance < sensor_range) {
                LandmarkObs         theLandmark;
                theLandmark.id       = aLandmark.id_i;
                theLandmark.x        = aLandmark.x_f;
                theLandmark.y        = aLandmark.y_f;
                predicted_landmarks.push_back(theLandmark);
            }
        }

        // 3 - find nearest landmarks for the transformed observations
        vector<LandmarkObs>        nearest_landmarks;
        nearest_landmarks = associateLandmarks(predicted_landmarks, transformed_observations); 

        // 4 - assign weights (probability) to the nearest landmarks
        double probability = 1.0;
        for (int m = 0; m < nearest_landmarks.size(); ++m) {
            double dx = transformed_observations.at(m).x - nearest_landmarks.at(m).x;
            double dy = transformed_observations.at(m).y - nearest_landmarks.at(m).y;

            // probability *= 1.0 / (2 * M_PI * std_x * std_y) * exp(-dx * dx / (2 * std_x * std_x)) * exp(-dy * dy/(2 * std_y*std_y));
            probability *= 1.0 / (std_xy_2PI) * exp(-dx * dx / (std_x_sq)) * exp(-dy * dy/(std_y_sq));
        
            a_particle.weight     = probability;
            weights[k]            = probability;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    discrete_distribution<int>  dist_particles(weights.begin(), weights.end());
    vector<Particle>            resampled_particles(num_particles);

    random_device rnd;
    default_random_engine rng(rnd());    // random number generator
    rng.seed(567);

    for(int k=0; k < num_particles; k++) {
        int j = dist_particles(rng);
        resampled_particles.at(k) = particles.at(j);
    }

    particles = resampled_particles;

    return;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
