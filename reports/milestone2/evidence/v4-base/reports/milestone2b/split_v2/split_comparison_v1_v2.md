# Split comparison: revision 1 (milestone2a) versus revision 2

Features and labels only; no model was run. Counts are rows and distinct feature-vector groups per class (a conflicting-label group counts toward each of its classes).

- Old manifest: `7275ede47c840d8820d6d5892ff6886af8ab23de9373f868a3e2d1c62c969cd2`
- New manifest: `842cbb882f5062946bdcab097f1370ddd796ff47439ad3581c493eec917b5d62` (rung 1, seed 20270)
- Rows preserved: True; labels unchanged: True (SHA-256 before `ec0dacdd849ac4d9422ba3a135934c32feb58c876e438ffd5cdc86261a8cf043`, after `ec0dacdd849ac4d9422ba3a135934c32feb58c876e438ffd5cdc86261a8cf043`)
- Shared groups old: {'train/validation': 0, 'train/test': 0, 'validation/test': 0}; new: {'train/validation': 0, 'train/test': 0, 'validation/test': 0}
- Classes with exactly one distinct train vector: old 15, new 0
- Old artifact hashes equal the pre-work snapshot: True

| Partition | Old rows | Old groups | New rows | New groups |
|---|---|---|---|---|
| train | 21925 | 5330 | 22412 | 7023 |
| validation | 4036 | 3327 | 3821 | 2467 |
| test | 4036 | 3296 | 3764 | 2463 |

Per class, rows / distinct groups, train | validation | test. `Floor` is the new rung floor (train/val/test); `Share` is the largest group's share of the class's train rows; `EffV` is the effective vector count 1 / sum(p_g^2) of the class's train rows (reported, not constrained). Weak = fewer than 10 distinct vectors in validation or test.

| Class | Name | Old tr | Old va | Old te | New tr | New va | New te | Floor | Share | EffV | Weak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Benign_Final | 641 / 552 | 235 / 235 | 235 / 235 | 607 / 607 | 255 / 205 | 249 / 210 | 340/205/205 | 0.002 | 607.00 |  |
| 1 | DDoS-ACK_Fragmentation | 656 / 1 | 228 / 159 | 227 / 163 | 818 / 163 | 141 / 80 | 152 / 80 | 107/65/65 | 0.802 | 1.55 |  |
| 2 | DDoS-HTTP_Flood | 640 / 412 | 235 / 206 | 236 / 216 | 647 / 450 | 229 / 191 | 235 / 193 | 278/167/167 | 0.306 | 10.56 |  |
| 3 | DDoS-ICMP_Flood | 1024 / 1 | 43 / 35 | 44 / 39 | 1067 / 44 | 24 / 15 | 20 / 16 | 25/15/15 | 0.960 | 1.09 |  |
| 4 | DDoS-ICMP_Fragmentation | 814 / 1 | 148 / 107 | 149 / 109 | 924 / 111 | 88 / 53 | 99 / 53 | 72/44/44 | 0.881 | 1.29 |  |
| 5 | DDoS-PSHACK_Flood | 990 / 1 | 61 / 41 | 60 / 39 | 1036 / 47 | 37 / 17 | 38 / 17 | 27/17/17 | 0.956 | 1.10 |  |
| 6 | DDoS-RSTFINFlood | 997 / 1 | 57 / 43 | 57 / 45 | 1049 / 53 | 32 / 18 | 30 / 18 | 29/18/18 | 0.950 | 1.11 |  |
| 7 | DDoS-SYN_Flood | 1020 / 1 | 45 / 27 | 46 / 31 | 1054 / 35 | 29 / 12 | 28 / 12 | 19/12/12 | 0.968 | 1.07 |  |
| 8 | DDoS-SlowLoris | 641 / 515 | 235 / 228 | 235 / 228 | 583 / 581 | 259 / 195 | 269 / 195 | 323/195/195 | 0.003 | 579.03 |  |
| 9 | DDoS-SynonymousIP_Flood | 1031 / 2 | 40 / 33 | 40 / 25 | 1064 / 35 | 20 / 13 | 27 / 12 | 20/12/12 | 0.968 | 1.07 |  |
| 10 | DDoS-TCP_Flood | 985 / 1 | 63 / 35 | 63 / 36 | 1026 / 42 | 42 / 15 | 43 / 15 | 24/15/15 | 0.960 | 1.08 |  |
| 11 | DDoS-UDP_Flood | 1008 / 1 | 52 / 31 | 51 / 33 | 1046 / 39 | 29 / 13 | 36 / 13 | 21/13/13 | 0.964 | 1.08 |  |
| 12 | DDoS-UDP_Fragmentation | 727 / 1 | 192 / 143 | 192 / 130 | 864 / 138 | 117 / 68 | 130 / 68 | 91/55/55 | 0.841 | 1.41 |  |
| 13 | DNS_Spoofing | 641 / 618 | 235 / 228 | 235 / 232 | 618 / 614 | 242 / 231 | 251 / 233 | 359/216/216 | 0.005 | 606.23 |  |
| 14 | DoS-HTTP_Flood | 640 / 511 | 236 / 227 | 235 / 224 | 587 / 576 | 274 / 193 | 250 / 193 | 320/193/193 | 0.020 | 479.23 |  |
| 15 | DoS-SYN_Flood | 930 / 1 | 91 / 45 | 90 / 43 | 983 / 53 | 66 / 18 | 62 / 18 | 29/18/18 | 0.946 | 1.12 |  |
| 16 | DoS-TCP_Flood | 1035 / 1 | 38 / 23 | 38 / 23 | 1061 / 27 | 25 / 10 | 25 / 10 | 15/10/10 | 0.975 | 1.05 |  |
| 17 | DoS-UDP_Flood | 803 / 1 | 154 / 68 | 154 / 52 | 873 / 71 | 114 / 25 | 124 / 25 | 40/25/25 | 0.920 | 1.18 |  |
| 18 | MITM-ArpSpoofing | 641 / 218 | 235 / 186 | 235 / 164 | 531 / 340 | 305 / 114 | 275 / 114 | 189/114/114 | 0.173 | 30.01 |  |
| 19 | Mirai-greeth_flood | 947 / 1 | 82 / 41 | 82 / 44 | 996 / 50 | 59 / 18 | 56 / 18 | 28/18/18 | 0.951 | 1.11 |  |
| 20 | Mirai-greip_flood | 975 / 1 | 68 / 41 | 68 / 33 | 1019 / 45 | 39 / 15 | 53 / 15 | 25/15/15 | 0.957 | 1.09 |  |
| 21 | Mirai-udpplain | 937 / 1 | 87 / 42 | 87 / 44 | 987 / 51 | 57 / 18 | 67 / 18 | 29/18/18 | 0.949 | 1.11 |  |
| 22 | Recon-HostDiscovery | 641 / 587 | 235 / 235 | 235 / 234 | 614 / 614 | 245 / 221 | 252 / 221 | 352/212/212 | 0.002 | 614.00 |  |
| 23 | Recon-OSScan | 641 / 508 | 235 / 230 | 235 / 228 | 581 / 578 | 292 / 194 | 238 / 194 | 322/194/194 | 0.003 | 575.06 |  |
| 24 | Recon-PingSweep | 640 / 602 | 235 / 235 | 236 / 235 | 616 / 616 | 245 / 222 | 250 / 234 | 357/215/215 | 0.002 | 616.00 |  |
| 25 | Recon-PortScan | 640 / 459 | 236 / 220 | 235 / 225 | 558 / 542 | 290 / 181 | 263 / 181 | 301/181/181 | 0.007 | 518.94 |  |
| 26 | VulnerabilityScan | 640 / 556 | 235 / 232 | 236 / 235 | 603 / 603 | 266 / 205 | 242 / 215 | 341/205/205 | 0.002 | 603.00 |  |
