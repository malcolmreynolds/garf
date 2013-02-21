#ifndef GARF_RANDOM_SEED_HPP
#define GARF_RANDOM_SEED_HPP

#include <random>
#include <iostream>
#include <fstream>

namespace garf { namespace util {

    typedef uint32_t seed_t;

    seed_t * random_seed_sequence(uint32_t sequence_length) {
        seed_t * seed_sequence = new seed_t[sequence_length];
        std::ifstream infile("/dev/random");

        for (uint32_t i = 0; i < sequence_length; i++) {
            infile >> seed_sequence[i];
        }
        return seed_sequence;
    }
}}



#endif