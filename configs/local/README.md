# Machine-local Lancet configuration

`lancet_email.env.example` is the tracked, secret-free template.
`lancet_email.env` is the ignored machine-local file to fill in. Never place a
real password, SMTP token, or App Password in the example, a shell command,
training metadata, logs, Agent Memory, or Handoff.

## Email fields

- `LANCET_EMAIL_ENABLED`: leave `false` while preparing credentials. A later
  explicitly authorized test can set it to `true` after notification code exists.
- `LANCET_SMTP_HOST`: the mail provider's SMTP server, such as
  `smtp.gmail.com` for Gmail.
- `LANCET_SMTP_PORT`: normally `587` with STARTTLS or `465` with direct SSL.
- `LANCET_SMTP_USER`: the SMTP account, generally the complete email address.
- `LANCET_SMTP_PASSWORD`: normally a provider-issued SMTP password or App
  Password, not the normal interactive mailbox login password.
- `LANCET_EMAIL_FROM`: sender address, normally equal to `LANCET_SMTP_USER`.
- `LANCET_EMAIL_TO`: mailbox that should receive Lancet notifications.
- `LANCET_SMTP_STARTTLS` / `LANCET_SMTP_SSL`: normally exactly one is `true`.
  A typical port-587 setup is `true/false`; port 465 is usually `false/true`.

No SMTP sender or training integration exists yet. Filling this file alone
does not send email or alter a running experiment.
