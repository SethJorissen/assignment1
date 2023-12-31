#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
    int ngram_;
    int log_num_buckets_;
    int num_hashes_;
    double learning_rate_;
    double bias_;
    std::vector<std::vector<double>> weights_;

public:
    /** Do not change the signature of the constructor! */
    PerceptronCountMin(int ngram, int num_hashes, int log_num_buckets,
                       double learning_rate)
        : BaseClf(0.0 /* set appropriate threshold */)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , num_hashes_(num_hashes)
        , learning_rate_(learning_rate)
        , bias_(0.0)
    {
        weights_.resize(num_hashes_, std::vector<double>((1 << log_num_buckets_), 0.0));
    }

    void update_(const Email& email) {
        int isSpam = email.is_spam() * 2 - 1;
        EmailIter allngrams(email, ngram_);
        std::vector<std::vector<double>> w (num_hashes_, std::vector<double>(1 << log_num_buckets_, 0.0));
        int bucket;
        std::vector<double> h (num_hashes_, 0.0);
        std::string_view ngram;
        while (allngrams) {
            ngram = allngrams.next();
            for (int i = 0; i < num_hashes_; i++) {
                bucket = get_bucket(ngram, i);
                ++w[i][bucket];
                h[i] += weights_[i][bucket];
            }
        }
        for (int i = 0; i < num_hashes_; i++) {
            h[i] = tanh(h[i]);
            scalarMulVector(w[i], learning_rate_ * (isSpam - h[i]) * (1 - h[i] * h[i]));
            vectorSub(weights_[i], w[i]);
        }
    }

    double predict_(const Email& email) const {
        EmailIter allngrams(email, ngram_);
        double h = 0.0;
        double h_i = 0.0;
        std::string_view ngram;
        while (allngrams) {
            ngram = allngrams.next();
            for (int i = 0; i < num_hashes_; i++) {
                h_i += weights_[i][get_bucket(ngram, i)];
            }
            h += h_i / num_hashes_;
            h_i = 0.0;
        }
        return tanh(h);
    }

private:
    size_t get_bucket(std::string_view ngram, int seed) const {
        return get_bucket(hash(ngram, seed));
    }

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
