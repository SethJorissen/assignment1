#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int seed_;
    int ngram_;
    int log_num_buckets_;
    int nSpam_;
    int nHam_;
    std::vector<int> counts_;

public:
    /** Do not change the signature of the constructor! */
    NaiveBayesFeatureHashing(int ngram, int log_num_buckets)
        : BaseClf(0.0 /* set appropriate threshold */)
        , seed_(0xfa4f8cc)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , nSpam_(1)
        , nHam_(1)
    {
        counts_.resize((1 << log_num_buckets_) * 2, 1);
    }

    void update_(const Email& email) {
        int isSpam = email.is_spam();
        EmailIter allngrams(email, ngram_);
        if (isSpam) {
            ++nSpam_;
        }
        else {
            ++nHam_;
        }
        while (allngrams)
        {
            ++counts_[get_bucket(allngrams.next(), isSpam)];
        }
    }

    double predict_(const Email& email) const {
        double result = log(nSpam_ / nHam_);
        EmailIter allngrams(email, ngram_);
        std::string_view ngram;
        while (allngrams)
        {
            ngram = allngrams.next();
            result += log(counts_[get_bucket(ngram, 1)] / counts_[get_bucket(ngram, 0)]);
        }
        return result / (1 + result);
    }

private:
    size_t get_bucket(std::string_view ngram, int is_spam) const {
        return get_bucket(hash(ngram, seed_), is_spam);
    }

    size_t get_bucket(size_t hash, int is_spam) const {
        hash &= (1 << log_num_buckets_) - 1;
        hash *= 2;
        hash += is_spam;
        return hash;
    }
};

} // namespace bdap