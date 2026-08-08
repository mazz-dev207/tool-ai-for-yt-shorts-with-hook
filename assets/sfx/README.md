# Smart Intro SFX

Pune aici propriile fișiere audio. Sunt acceptate: `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`.

Tool-ul citește tag-urile direct din numele fișierului. Separă cuvintele prin `_`, `-` sau spații.

## Exemple recomandate

- `whoosh_fast_reveal_01.mp3`
- `whoosh_fast_surprise_02.mp3`
- `impact_reveal_surprise_01.mp3`
- `impact_gaming_clutch_01.mp3`
- `impact_gaming_win_02.mp3`
- `whoosh_soft_storytelling_01.mp3`
- `whoosh_soft_emotional_01.mp3`
- `pop_funny_comedy_01.mp3`
- `whoosh_funny_light_01.mp3`

## Reguli automate

- surprise / reveal / twist -> whoosh + impact
- gaming / clutch / win -> impact
- storytelling -> whoosh soft/subtle
- emotional -> evită impact/hit/boom/hard/bass; preferă whoosh soft
- funny / comedy / meme -> pop sau whoosh

Dacă există mai multe fișiere potrivite, tool-ul alege random dintre cele mai bune potriviri.

Dacă folderul este gol sau nu există un efect potrivit, pipeline-ul continuă normal fără SFX.
