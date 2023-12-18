#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {
    int ngram_;
    int log_num_buckets_;
    int num_hashes_;
    int nSpam_;
    int nHam_;
    int nSpamGrams_;
    int nHamGrams_;
    std::vector<std::vector<int>> counts_;

public:
    NaiveBayesCountMin(int ngram, int num_hashes, int log_num_buckets)
        : BaseClf(0.0 /* set appropriate threshold */)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , num_hashes_(num_hashes)
        , nSpam_(1)
        , nHam_(1)
        , nSpamGrams_(1)
        , nHamGrams_(1)
    {
        counts_.resize(num_hashes, std::vector<int>((1 << log_num_buckets_) * 2, 1));
    }

    void update_(const Email &email) {
        int isSpam = email.is_spam();
        EmailIter allngrams(email, ngram_);
        if (isSpam) {
            ++nSpam_;
            while (allngrams)
            {
                for (int i = 0; i < num_hashes_; i++) {
                    ++counts_[i][get_bucket(allngrams.next(), isSpam, i)];
                }
                ++nSpamGrams_;
            }
        }
        else {
            ++nHam_;
            while (allngrams)
            {
                for (int i = 0; i < num_hashes_; i++) {
                    ++counts_[i][get_bucket(allngrams.next(), isSpam, i)];
                }
                ++nHamGrams_;
            }
        }
    }

    double predict_(const Email& email) const {
        double result = log((double)nSpam_ / (double)nHam_);
        EmailIter allngrams(email, ngram_);
        std::string_view ngram;
        while (allngrams)
        {
            ngram = allngrams.next();
            result += log(((double)count(ngram, 1) / (double)nSpamGrams_) / ((double)count(ngram, 0) / (double)nHamGrams_));
        }
        return result / (1 + result);
    }

private:
    int count(std::string_view ngram, int is_spam) const {
        int test;
        int min = counts_[0][get_bucket(ngram, is_spam, 0)];
        for (int i = 1; i < num_hashes_; i++) {
            test = counts_[i][get_bucket(ngram, is_spam, i)];
            if (test < min) {
                min = test;
            }
        }
        return min;
    }

    size_t get_bucket(std::string_view ngram, int is_spam, int seed) const {
        return get_bucket(hash(ngram, seed), is_spam);
    }

    size_t get_bucket(size_t hash, int is_spam) const {
        hash &= (1 << log_num_buckets_) - 1;
        hash *= 2;
        hash += is_spam;
        return hash;
    }
};

} // namespace bdap
