- [MTS (Minimum Time of Stay) (1s is recommended for simulations)](https://www.yumpu.com/en/document/read/25629328/3gpp-tr-36839-v1110-2012-12/11)
- [A3 RSRP](https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/18.04.00_60/ts_138331v180400p.pdf)


## Wave Utils
- [Log-Distance Path Loss Model(pd0 and path_loss)](https://www.idc-online.com/technical_references/pdfs/electronic_engineering/Log_Distance_Path_Loss_or_Log_Normal_Shadowing_Model.pdf)
- [Thermal Noise (-174 dBm/Hz) (3GPP TR 25.942 / ETSI TR 125 942)](https://www.etsi.org/deliver/etsi_tr/125900_125999/125942/11.00.00_60/tr_125942v110000p.pdf)
- RSRP, RSSI, and RSRQ Physical Layer Definitions:
   - [LTE Definitions (TS 36.214)](https://www.etsi.org/deliver/etsi_ts/136200_136299/136214/14.03.00_60/ts_136214v140300p.pdf)
   - [5G NR Definitions (TS 38.215)](https://www.etsi.org/deliver/etsi_ts/138200_138299/138215/15.02.00_60/ts_138215v150200p.pdf)
- Measurement Report Mapping
    - [LTE RSRP & RSRQ Mapping (TS 36.133)](https://www.etsi.org/deliver/etsi_ts/136100_136199/136133/08.02.00_60/ts_136133v080200p.pdf)
    - [5G NR RSRP & RSRQ Mapping (TS 38.133)](https://www.etsi.org/deliver/etsi_ts/138100_138199/138133/15.03.00_60/ts_138133v150300p.pdf)
- Shadow Fading (Gudmundson Autocorrelation Model):
    - [3GPP TR 38.901 V17.0.0, Section 7.4.4 — "Large scale parameter generation"](https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/)
    - Table 7.5-6 (UMi scenario): shadow fading std dev (LOS: 4.0 dB, NLOS: 7.82 dB), decorrelation distance (LOS: 37m, NLOS: 50m)
    - Original paper: M. Gudmundson, "Correlation model for shadow fading in mobile radio systems," Electronics Letters, vol. 27, no. 23, pp. 2145–2146, Nov. 1991.
- Fast Fading (Rayleigh / Rician Envelope Models):
    - [3GPP TR 38.901 V17.0.0, Section 7.5 — "Fast fading model"](https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/)
    - Table 7.5-6 (UMi-LOS): Rician K-factor = 9 dB
    - NLOS: Rayleigh fading (no dominant path, K=0)
    - Textbook reference: A. Goldsmith, "Wireless Communications," Cambridge University Press, 2005 — Chapter 3.3 (Rayleigh and Rician fading envelope distributions)

[Resource Block Mapping Configurations](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/15.02.00_60/ts_13810101v150200p.pdf)


## System-Level Abstraction Notes

This simulation is a **system-level abstraction** of the 5G NR physical layer, not a full PHY emulator. The following simplifying assumptions were made:

### 1. LOS/NLOS Hard Cutoff (`__los_threshold = 5`)

**3GPP Reality:** In 3GPP TR 38.901, Line-of-Sight (LOS) is a probability function based on distance (e.g., at 18 meters, there is a 100% chance of LOS; at 100 meters, it drops to ~20%).

**Our Abstraction:** A hard 5-meter cutoff was applied.

**Justification:** To simulate a highly dense Urban Canyon environment where vehicles are almost perpetually in Non-Line-of-Sight (NLOS) conditions relative to macro-cells, a strict LOS threshold of 5 meters was applied.

### 2. Omnidirectional Antennas vs. Beamforming

**3GPP Reality:** 5G NR uses Massive MIMO and active beamforming to track UEs.

**Our Abstraction:** The `G_tx` parameter acts as a static isotropic (omnidirectional) gain.

**Justification:** Antenna radiation patterns were modeled isotropically, with complex beamforming gains abstracted into a constant $G_{tx}$ to isolate the evaluation of the handover algorithm from beam-tracking mechanics.

### 3. Standard Log-Distance Path Loss

**3GPP Reality:** Full 3GPP models use dual-slope path loss with specific breakpoint distances.

**Our Abstraction:** The standard log-distance path loss model was used.

**Justification:** Large-scale path loss was computed using the standard Log-Distance model, which provides a widely accepted balance of physical realism and computational efficiency necessary for the high-throughput Monte Carlo simulations required by Reinforcement Learning.