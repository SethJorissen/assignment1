#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int ngram_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;

    int seed_;

public:
    /** Do not change the signature of the constructor! */
    PerceptronFeatureHashing(int ngram, int log_num_buckets, double learning_rate)
        : BaseClf(0.0 /* set appropriate threshold */)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0xa738cc)
    {
        // set all weights to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email) {
        int isSpam = email.is_spam() * 2 + 1;
        EmailIter allngrams(email, ngram_);
        std::vector<int> w (1 << log_num_buckets_, 0);
        int bucket;
        double h = 0.0;
        for (allngrams) {
            bucket = get_bucket(allngrams.next());
            ++w[bucket];
            h += weights_[bucket];
        }
        h = tanh(h);
        w = learning_rate_ * (isSpam - h) * (1 - h * h) * w;
        weights_ -= w;
    }

    double predict_(const Email& email) const {
        EmailIter allngrams(email, ngram_);
        double h = 0.0;
        for (allngrams) {
            h += weights_[get_bucket(allngrams.next())];
        }
        return tanh(h);
    }


private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const {
        hash &= pow(2, log_num_buckets) - 1;
        return hash;
    }
};

} // namespace bdap