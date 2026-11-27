| Rival | Winkler del rival | Δ (PADRE − rival) | δ de Cliff (estaciones) | IC 95 % de Δ en la serie horaria (bootstrap estacionario) | Estadístico DM-HLN | p Wilcoxon (Holm) | p DM-HLN (Holm) | p bootstrap (Holm) | p máx. (Holm) | Conclusión |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | `3.153` | `-0.1621` | `-0.088` (despreciable) | [`-0.1765`, `-0.1343`] | `-18.39` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| TCN | `3.163` | `-0.1718` | `-0.095` (despreciable) | [`-0.1905`, `-0.1423`] | `-19.03` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| Ridge | `3.166` | `-0.1749` | `-0.091` (despreciable) | [`-0.1989`, `-0.1505`] | `-17.15` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| KAN | `3.171` | `-0.1796` | `-0.092` (despreciable) | [`-0.2069`, `-0.1457`] | `-16.66` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| Informer | `3.171` | `-0.1802` | `-0.095` (despreciable) | [`-0.1981`, `-0.1523`] | `-20.11` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| PatchTST | `3.176` | `-0.1847` | `-0.092` (despreciable) | [`-0.2069`, `-0.1552`] | `-18.58` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| GRU | `3.179` | `-0.1876` | `-0.104` (despreciable) | [`-0.2036`, `-0.1475`] | `-16.63` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| DLinear | `3.187` | `-0.1959` | `-0.092` (despreciable) | [`-0.2258`, `-0.1662`] | `-17.42` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| Gradient boosting | `3.193` | `-0.2016` | `-0.096` (despreciable) | [`-0.2356`, `-0.1629`] | `-15.72` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| STGCN | `3.198` | `-0.2069` | `-0.102` (despreciable) | [`-0.2248`, `-0.1689`] | `-18.83` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| Chronos | `3.891` | `-0.8997` | `-0.255` (pequeño) | [`-1.0250`, `-0.6911`] | `-21.07` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
| Moirai | `4.610` | `-1.6188` | `-0.531` (grande) | [`-1.7939`, `-1.4297`] | `-38.95` | `< 0.001` | `< 0.001` | `0.001` | `0.001` | rechaza; a favor de PADRE |
