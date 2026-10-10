# Security Policy

## Supported Versions

SovereignGrad is currently in early development (v0.1.x). Security fixes are applied to the `main` branch only.

| Version | Supported |
| ------- | --------- |
| 0.1.x (main) | ✅ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

To report a security vulnerability, please use **[GitHub's private vulnerability reporting](https://github.com/bharqav/sovereigngrad/security/advisories/new)**.

Include the following:
- Description of the vulnerability
- Steps to reproduce
- Affected version(s)
- Potential impact

You will receive an acknowledgment within 72 hours. We will work with you to understand and address the issue before public disclosure.

## Security Scope

SovereignGrad is a C++ library with no network-facing interfaces, authentication systems, or data persistence. The primary security considerations are:

- **Memory safety**: The codebase uses raw pointers in performance-critical paths. The ASan/UBSan CI job validates these paths.
- **Untrusted model files**: Loading `.sgrd` or `.safetensors` files from untrusted sources is not safe - the parsers do not perform bounds-checked deserialization against adversarial inputs. Do not load checkpoint files from untrusted parties without validation.
- **Integer overflow**: Large tensor shapes could theoretically produce integer overflows in `compute_numel`. Shape inputs from untrusted sources should be range-checked by the caller.

## What is NOT in scope

- Python benchmarking scripts (read-only analytics, no network access)
- CMake build configuration
- Documentation
