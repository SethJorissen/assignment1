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
        int isSpam = email.is_spam() * 2 - 1;
        std::cout << "IsSpam: " << isSpam << std::endl;
        EmailIter allngrams(email, ngram_);
        std::vector<double> w (1 << log_num_buckets_, 0.0);
        int bucket;
        double h = 0.0;
        while (allngrams) {
            bucket = get_bucket(allngrams.next());
            ++w[bucket];
            h += weights_[bucket];
        }
        //h = tanh(h);
        std::cout << "Predict: " << h << std::endl;
        //scalarMulVector(w, learning_rate_ * (isSpam - h) * (1 - h * h));
        scalarMulVector(w, learning_rate_ * (isSpam - h));
        std::cout << "update: [";
        for (int i; i < (1 << log_num_buckets_); i++) {
            std::cout << w[i] << ", ";
        }
        std::cout << "]" << std::endl;
        vectorSub(weights_, w);
    }

    double predict_(const Email& email) const {
        EmailIter allngrams(email, ngram_);
        double h = 0.0;
        while (allngrams) {
            h += weights_[get_bucket(allngrams.next())];
        }
        return h;
    }


private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const {
        hash &= (1 << log_num_buckets_) - 1;
        return hash;
    }

    void scalarMulVector(std::vector<double>& v, double k) {
        std::transform(v.begin(), v.end(), v.begin(), [k](double& c) { return c * k; });
    }

    void vectorSub(std::vector<double>& v1, std::vector<double>& v2) {
        std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), [](double a, double b) { return a - b; });
    }
};

} // namespace bdap
