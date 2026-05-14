#ifndef TASSEL_CORE_LOSS_FUCTION_LOSS_FUCTION_BASE_H_
#define TASSEL_CORE_LOSS_FUCTION_LOSS_FUCTION_BASE_H_

#include <cmath>
#include <variant>

namespace tassel_core {

// ── Template-specialized loss function types (no virtual functions) ──────────
//
// Each loss type computes:
//   rho(s)      – robust cost (used for error tracking)
//   rho_prime(s) – first derivative w.r.t. s
//   weight(s)   – IRLS weight w = rho'(s) / s (used for Jacobian scaling)
//
// where s = ||residual|| (the 2-norm of the 2D reprojection residual).

// ── Trivial (least-squares) ──────────────────────────────────────────────────

struct TrivialLoss {
    // For template API compatibility
    TrivialLoss() = default;

    static constexpr double rho(double s) { return 0.5 * s * s; }

    static constexpr double rho_prime(double s) { return s; }

    // IRLS weight: rho'(s) / s
    static constexpr double weight(double /*s*/) { return 1.0; }
};

// ── Huber ────────────────────────────────────────────────────────────────────

struct HuberLoss {
    double delta;

    explicit HuberLoss(double d = 1.345) : delta(d) {}

    double rho(double s) const {
        if (s <= delta) return 0.5 * s * s;
        return delta * (s - 0.5 * delta);
    }

    double rho_prime(double s) const {
        if (s <= delta) return s;
        return delta;
    }

    double weight(double s) const {
        if (s <= delta) return 1.0;
        return delta / s;
    }
};

// ── Cauchy ──────────────────────────────────────────────────────────────────

struct CauchyLoss {
    double c;

    explicit CauchyLoss(double c_ = 2.3849) : c(c_) {}

    double rho(double s) const { return 0.5 * c * c * std::log1p((s * s) / (c * c)); }

    double rho_prime(double s) const { return s / (1.0 + (s * s) / (c * c)); }

    double weight(double s) const { return 1.0 / (1.0 + (s * s) / (c * c)); }
};

// ── Tukey (biweight) ────────────────────────────────────────────────────────

struct TukeyLoss {
    double c;

    explicit TukeyLoss(double c_ = 4.6851) : c(c_) {}

    double rho(double s) const {
        if (s >= c) return c * c / 6.0;
        double t = s * s / (c * c);
        return (c * c / 6.0) * (1.0 - std::pow(1.0 - t, 3));
    }

    double rho_prime(double s) const {
        if (s >= c) return 0.0;
        double t = s * s / (c * c);
        return s * (1.0 - t) * (1.0 - t);
    }

    double weight(double s) const {
        if (s >= c) return 0.0;
        double t = s * s / (c * c);
        double w = 1.0 - t;
        return w * w;
    }
};

// ── Runtime-polymorphic loss (std::variant, no virtual functions) ────────────

using LossVariant = std::variant<TrivialLoss, HuberLoss, CauchyLoss, TukeyLoss>;

/// Compute the robust cost rho(s).
inline double computeRho(const LossVariant& loss, double s) {
    return std::visit([s](const auto& l) { return l.rho(s); }, loss);
}

/// Compute the IRLS weight rho'(s) / s.
inline double computeWeight(const LossVariant& loss, double s) {
    return std::visit([s](const auto& l) { return l.weight(s); }, loss);
}

}  // namespace tassel_core
#endif  // TASSEL_CORE_LOSS_FUCTION_LOSS_FUCTION_BASE_H_
