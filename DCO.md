# Developer Certificate of Origin (DCO) v1.1
## For the Feistback Engine Project (EUPL-1.2)

To ensure a clear chain of authorship and legal compliance with the **European Union Public Licence (EUPL-1.2)** (specifically Article 5 and Article 6, *Chain of Authorship*), the Feistback Engine project uses the standard Developer Certificate of Origin (DCO).

The DCO is a lightweight mechanism that avoids complex Contributor License Agreement (CLA) management systems while providing full legal clarity for open-source contributors and project maintainers.

---

### Developer Certificate of Origin 1.1

By making a contribution to this project, I certify that:

1. **Original Work:** The contribution was created in whole or in part by me and I have the right to submit it under the European Union Public Licence (EUPL-1.2) as indicated in the repository; or

2. **Permissive/Compatible Upstream:** The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open-source license compatible with EUPL-1.2 (as defined in Article 5 and the EUPL Appendix, including GPLv2/v3, AGPLv3, LGPL, MPL, Apache 2.0, or MIT), and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the EUPL-1.2; or

3. **Direct Authorization:** The contribution was provided directly to me by some other person who certified (1), (2) or (3) and I have not modified it; and

4. **Public Record:** In the case of each of (1), (2), or (3), I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project, the EUPL-1.2, and the open-source licensing involved.

---

### How to Sign Off on Contributions

Contributors sign off on their work by adding a `Signed-off-by` trailer line to every commit message:

```text
Signed-off-by: Random J Developer <random@example.com>
```

Git automates this using the `-s` flag:
```bash
git commit -s -m "feat(scope): concise description"
```

For multiline commits (following the project's Tier 1 / Tier 2 commit standards described in `AGENTS.md` and `CONTRIBUTING.md`), simply ensure the `Signed-off-by:` trailer is included alongside any architectural metadata trailers (`Invariant:`, `Hypothesis:`, `Rejected:`).
