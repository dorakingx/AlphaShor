#!/usr/bin/env python3
import random
import math
import json

# ---------------------
# Helper Functions
# ---------------------

def is_prime(n):
    """Trial division to test primality (suitable for small n)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_candidate_prime(bit_length, max_attempts=1000000):
    """
    Randomly search for a prime with exactly 'bit_length' bits that:
      - is greater than 3 (unsure why but Vlad recommended it).
      - is not 7 (to avoid a singular curve since our curve is y^2 = x^3 + 7).
      - satisfies p ≡ 1 mod 3 (to ensure the curve is not supersingular).
    """
    lower = 1 << (bit_length - 1)
    upper = (1 << bit_length) - 1
    for _ in range(max_attempts):
        candidate = random.randint(lower, upper)
        if candidate > 3 and candidate != 7 and candidate % 3 == 1 and is_prime(candidate):
            return candidate
    return None

def find_all_candidate_primes(bit_length):
    """
    Return a list of all primes with exactly 'bit_length' bits that:
      - are greater than 3,
      - are not 7 (to avoid the singular curve for y^2 = x^3 + 7),
      - and satisfy p ≡ 1 mod 3.
    """
    lower = 1 << (bit_length - 1)
    upper = (1 << bit_length) - 1
    candidates = []
    for candidate in range(lower, upper + 1):
        if candidate > 3 and candidate != 7 and candidate % 3 == 1 and is_prime(candidate):
            candidates.append(candidate)
    return candidates

def mod_inv(a, p):
    """Modular inverse of a modulo p (p assumed prime)."""
    return pow(a, -1, p)

def point_add(P, Q, p):
    """
    Add two points P and Q on the elliptic curve over F_p.
    Points are represented as tuples (x, y) and the identity is None.
    """
    if P is None:
        return Q
    if Q is None:
        return P
    # Check for vertical tangency (result is the point at infinity).
    if P[0] == Q[0]:
        # If y coordinates add to 0 mod p, P + Q = 0.
        if (P[1] + Q[1]) % p == 0:
            return None
        # Point doubling:
        # Note: curve parameter a = 0 for our curve.
        s = (3 * P[0] * P[0]) * mod_inv(2 * P[1], p) % p
    else:
        s = (Q[1] - P[1]) * mod_inv(Q[0] - P[0], p) % p
    x_r = (s * s - P[0] - Q[0]) % p
    y_r = (s * (P[0] - x_r) - P[1]) % p
    return (x_r, y_r)

def scalar_mult(k, P, p):
    """Compute k * P by double-and-add; P is a point on the curve over F_p."""
    R = None  # The identity element
    Q = P
    while k > 0:
        if k & 1:
            R = point_add(R, Q, p)
        Q = point_add(Q, Q, p)
        k //= 2
    return R

def tonelli_shanks(n, p):
    """
    Solve for a square root of n modulo p (if one exists) using the Tonelli-Shanks algorithm.
    Returns one square root; the other is p - result.
    """
    # Check existence using Euler's criterion.
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    # Shortcut when p ≡ 3 mod 4.
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    # Write p - 1 as Q * 2^S with Q odd.
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    # Find a quadratic non-residue z.
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)
    while t != 1:
        t2i = t
        i = 0
        while t2i != 1:
            t2i = (t2i * t2i) % p
            i += 1
            if i == M:
                return None
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        R = (R * b) % p
        t = (t * c) % p
    return R

def count_points(p):
    """
    Deterministically count the points on the curve y^2 = x^3 + 7 over F_p.
    (Counts the point at infinity as well.)
    WARNING: This method loops over all x in F_p (O(p)) and may be very slow for large p.
    """
    count = 1  # Start with the point at infinity.
    for x in range(p):
        rhs = (x * x * x + 7) % p
        if rhs == 0:
            count += 1  # Only solution y = 0.
        else:
            # Use Euler's criterion to test if rhs is a quadratic residue.
            if pow(rhs, (p - 1) // 2, p) == 1:
                count += 2  # Two solutions: y and p - y.
    return count

def factorize(n):
    """Return a dictionary of prime factors of n with their exponents (trial division)."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = 1
    return factors

def pick_random_point(p):
    """
    Randomly pick a point on the curve y^2 = x^3 + 7 over F_p.
    Returns a tuple (x, y).
    """
    while True:
        x = random.randint(0, p - 1)
        rhs = (x * x * x + 7) % p
        if rhs == 0:
            # y = 0 is the only solution.
            return (x, 0)
        # Check if rhs is a quadratic residue.
        if pow(rhs, (p - 1) // 2, p) == 1:
            y = tonelli_shanks(rhs, p)
            if y is None:
                continue
            # Randomly choose one of the two square roots.
            if random.choice([True, False]):
                y = p - y
            return (x, y)

# ---------------------
# Main Function
# ---------------------

## Not used
def random_checking(upper_bound):
    # Fix the seed for reproducibility.
    seed = 536
    random.seed(seed)

    # Loop over bit sizes from 1 to upper_bound.
    successful_runs = 0
    for bit_length in range(1, upper_bound):
        print(f"\n--- Bit size {bit_length} ---")

        # Generate a suitable prime p.
        p = generate_candidate_prime(bit_length)
        if p is None:
            print(f"No suitable prime found for bit size {bit_length}.")
            continue

        # WARNING: Point counting using a loop over F_p is O(p) and may be very slow for large p.
        if p > 10000:
            print("p is large; point counting may be slow. (This may take a while...)")
        order = count_points(p)

        # Factorize the order to find its prime factors.
        factors = factorize(order)
        n = max(factors.keys())  # Choose the largest prime factor as subgroup order.
        h = order // n

        # Ensure the subgroup is "large" i.e that is has a small cofactor.
        if h > 3:
            print("Skipping candidate: subgroup cofactor is too large (n is too small).")
            continue

        # Pick a random point on the curve.
        P = pick_random_point(p)
        # “Project” P into the subgroup by multiplying by the cofactor h.
        G = scalar_mult(h, P, p)
        if G is None:
            print("Failed to generate a non-identity subgroup point. Trying another bit size.")
            continue

        # Verify that the order of G is exactly n: n * G should be the identity (None).
        if scalar_mult(n, G, p) is not None:
            print("Warning: The chosen generator G does not have order n. Skipping this bit size.")
            continue

        # Generate a key pair:
        # Choose a random private key d in [1, n - 1].
        d = random.randint(1, n - 1)
        # Compute the public key Q = d * G.
        Q = scalar_mult(d, G, p)

        successful_runs += 1

        print(f"Prime p: {p}")
        print(f"Curve order (number of points #E): {order}")
        print(f"Subgroup order n (largest prime factor of #E): {n}")
        print(f"Cofactor h: {h}")
        print(f"Generator point G: {G}")
        print(f"Private key d: {d}")
        print(f"Public key Q: {Q}")

    print("All done!")
    print(f"Seed: {seed}")
    print(f"Total successful runs: {successful_runs}")

def strict_check(upper_bound):
    # Fix the seed for reproducibility.
    seed = 536
    random.seed(seed)

    # Open the output file.
    output_filename = "successful_curves.txt"
    output_json = "successful_curves.json"
    results = []
    with open(output_filename, "w") as f_out:
        f_out.write("Q Day Curves\n")
        f_out.write(f"Seed for reproducibility: {seed}\n")
        f_out.write("=" * 40 + "\n\n")

        successful_runs = 0
        for bit_length in range(1, upper_bound):
            header = f"\n--- Bit size {bit_length} ---\n"
            print(header)
            f_out.write(header)

            # Get the full list of candidate primes for this bit length.
            candidate_primes = find_all_candidate_primes(bit_length)
            if not candidate_primes:
                msg = f"No candidate primes found for bit size {bit_length}.\n"
                print(msg)
                f_out.write(msg)
                continue

            # Try each candidate systematically until one yields a valid curve.
            found = False
            for p in candidate_primes:
                msg = f"\nTrying candidate prime: {p}\n"
                print(msg, end="")

                if p > 10000:
                    msg = "p is large; point counting may be slow. (This may take a while...)\n"
                    print(msg, end="")
                order = count_points(p)
                factors = factorize(order)
                n = max(factors.keys())  # Largest prime factor as subgroup order.
                h = order // n
                msg = f"Candidate {p}: Curve order = {order}, Subgroup order n = {n}, Cofactor h = {h}\n"
                print(msg, end="")

                # Ensure the subgroup is "large" (i.e. h is small).
                if h > 2:
                    msg = f"Skipping candidate {p}: subgroup cofactor too large (h = {h}).\n"
                    print(msg, end="")
                    continue

                # Pick a random point on the curve.
                P = pick_random_point(p)
                # Project P into the subgroup by multiplying by the cofactor.
                G = scalar_mult(h, P, p)
                if G is None:
                    msg = f"Candidate {p} failed to generate a non-identity subgroup point.\n"
                    print(msg, end="")
                    continue

                # Verify that the order of G is exactly n.
                if scalar_mult(n, G, p) is not None:
                    msg = f"Candidate {p} generated a G with incorrect order. Skipping.\n"
                    print(msg, end="")
                    continue

                # Generate key pair:
                d = random.randint(1, n - 1)
                Q = scalar_mult(d, G, p)

                if Q is None:
                    msg = f"Candidate {p} failed to generate a valid public key.\n"
                    print(msg, end="")
                    continue

                # This candidate is successful.
                success_msg = f"Candidate {p} successful!\n"
                print(success_msg, end="")
                f_out.write(f"Bit size: {bit_length}\n")
                f_out.write(f"Prime p: {p}\n")
                f_out.write(f"Curve order (#E): {order}\n")
                f_out.write(f"Subgroup order n: {n}\n")
                f_out.write(f"Cofactor h: {h}\n")
                f_out.write(f"Generator point G: {G}\n")
                f_out.write(f"Private key d: {d}\n")
                f_out.write(f"Public key Q: {Q}\n")

                results.append({
                    "bit_length": bit_length,
                    "prime": p,
                    "curve_order": order,
                    "subgroup_order": n,
                    "cofactor": h,
                    "generator_point": list(G),
                    "private_key": d,
                    "public_key": list(Q),
                })

                successful_runs += 1
                found = True
                # Optionally break after the first valid candidate per bit size.
                break

            if not found:
                msg = f"No valid curve found for bit size {bit_length}.\n"
                print(msg, end="")
                f_out.write(msg)

    # Write out the JSON file
    with open(output_json, "w") as jf:
        json.dump(results, jf, indent=2)

        print("\nAll done!")
        print(f"Total successful runs: {successful_runs}")

if __name__ == "__main__":
    # random_checking(22)
    strict_check(22)
