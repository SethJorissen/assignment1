#include "email.hpp"

namespace bdap {

    struct Accuracy {
        int n = 0;
        int correct = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
            ++n;
            correct += static_cast<int>(lab == pred);
        }

        double get_accuracy() const { return static_cast<double>(correct) / n; }
        double get_error() const { return 1.0 - get_accuracy(); }

        double get_score() const { return get_accuracy(); }
    };

    struct precision {
        int pr_pos = 0;
        int tr_pos = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
            if (pred) {
                ++pr_pos;
                bool lab = email.is_spam();
                if (lab) {
                    ++tr_pos;
                }
            }
        }

        double get_precision() const { return tr_pos / pr_pos; }
        double get_error() const { return 1.0 - get_precision(); }

        double get_score() const { return get_precision(); }
    };

    struct recall {
        int pos = 0;
        int tr_pos = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            if (lab) {
                ++pos;
                double pr = clf.predict(email);
                bool pred = clf.classify(pr);
                if (pred) {
                    ++tr_pos;
                }
            }
        }

        double get_precision() const { return tr_pos / pos; }
        double get_error() const { return 1.0 - get_precision(); }

        double get_score() const { return get_precision(); }
    };

    struct ConfusionMatrix {

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, Email& emails) {

        }

    };


} // namespace bdap
