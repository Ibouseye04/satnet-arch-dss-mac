# Alex Windows Tailscale SSH Setup

Last reviewed: 2026-06-28

This guide transitions Alex from VS Code tunneling to SSH over Tailscale so Ibrahim can connect to Alex's Windows 11 Home machine without router port forwarding or exposing SSH to the public internet.

## Target Architecture

- Ibrahim owns the tailnet and signs in as `iseye@mearas.us`.
- Alex uses his own Tailscale login and joins Ibrahim's tailnet as a regular member.
- Alex's Windows PC runs the normal Microsoft OpenSSH Server.
- Ibrahim connects to Alex's Windows PC over its Tailscale IPv4 address or MagicDNS name.
- Tailscale SSH is not used on Alex's Windows machine because Tailscale SSH's server component is currently limited to Linux and macOS open source `tailscale + tailscaled` CLI devices.

## Machine Notes

Alex's machine, from the provided photo:

- OS: Windows 11 Home
- CPU: AMD Ryzen 7 8700F
- RAM: 32 GB DDR5-5200
- GPU: NVIDIA GeForce RTX 4060
- Network: 2.5 GbE LAN, Wi-Fi 6, Bluetooth

This is a normal x64 Windows 11 PC. Use the standard Tailscale Windows installer.

## Accounts Needed

Ibrahim:

- Tailscale owner/admin account: `iseye@mearas.us`
- A local SSH private key on Ibrahim's computer
- The matching SSH public key to send to Alex

Alex:

- His own Tailscale account/login email
- Access to Ibrahim's Tailscale invite
- A Windows local account that Ibrahim will SSH into

Do not have Alex log in with Ibrahim's Tailscale credentials. Add Alex to the tailnet with his own identity so access can be revoked cleanly later.

## Ibrahim: Prepare the SSH Public Key

On Ibrahim's machine, check for an existing public key:

```bash
ls ~/.ssh/*.pub
```

If no key exists, create one:

```bash
ssh-keygen -t ed25519 -C "iseye@mearas.us alex-windows-tailscale"
```

Print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Send only the `.pub` line to Alex. Never send the private key file.

## Ibrahim: Invite Alex to the Tailnet

1. Sign in to the Tailscale admin console as `iseye@mearas.us`.
2. Open the Users page.
3. Choose **Invite external users**.
4. Enter Alex's email address.
5. Assign the **Member** role unless Alex needs admin access.
6. Send the invite.

The Tailscale Personal plan currently allows multiple users in one tailnet, so this should fit a small friend/collaboration setup.

## Alex: Install and Sign In to Tailscale

1. Download Tailscale for Windows from the official Tailscale download page.
2. Run the Windows installer.
3. Open the Tailscale tray icon.
4. Select **Log in**.
5. Sign in with Alex's own Tailscale identity.
6. Accept Ibrahim's tailnet invite if prompted.
7. Confirm the Tailscale tray icon shows that Tailscale is connected.

To verify from PowerShell:

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" status
& "$env:ProgramFiles\Tailscale\tailscale.exe" ip -4
```

Alex should send Ibrahim:

- The Tailscale IPv4 address, usually `100.x.y.z`
- The Windows username Ibrahim should SSH into
- The machine name shown in the Tailscale admin console, if available

## Alex: Enable Microsoft OpenSSH Server

Open PowerShell as Administrator and run:

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
Get-Service sshd
```

Microsoft's installer normally creates a Windows Firewall rule named `OpenSSH-Server-In-TCP` for inbound TCP/22.

Restrict that rule so it only accepts inbound SSH from Tailscale's default IPv4 range:

```powershell
Set-NetFirewallRule -Name OpenSSH-Server-In-TCP -RemoteAddress 100.64.0.0/10 -Profile Any
(Get-NetFirewallRule -Name OpenSSH-Server-In-TCP | Get-NetFirewallAddressFilter).RemoteAddress
```

If the rule does not exist, create it explicitly:

```powershell
New-NetFirewallRule `
  -Name OpenSSH-Server-In-TCP `
  -DisplayName "OpenSSH SSH Server (sshd)" `
  -Enabled True `
  -Direction Inbound `
  -Protocol TCP `
  -Action Allow `
  -LocalPort 22 `
  -RemoteAddress 100.64.0.0/10 `
  -Profile Any
```

Do not configure router port forwarding for port 22. Tailscale does not require it.

## Alex: Install Ibrahim's Public SSH Key

Use a standard non-admin Windows account if possible. It keeps remote access scoped and avoids Windows' special administrator key path.

Replace `AlexWindowsUser` with the actual Windows username and replace the public key placeholder with Ibrahim's full public key line.

```powershell
$sshDir = "C:\Users\AlexWindowsUser\.ssh"
New-Item -ItemType Directory -Force -Path $sshDir
notepad "$sshDir\authorized_keys"
```

Paste Ibrahim's public key into `authorized_keys`, save, and close Notepad.

Set conservative permissions:

```powershell
icacls "C:\Users\AlexWindowsUser\.ssh" /inheritance:r /grant "AlexWindowsUser:F"
icacls "C:\Users\AlexWindowsUser\.ssh\authorized_keys" /inheritance:r /grant "AlexWindowsUser:F"
Restart-Service sshd
```

If Ibrahim must SSH into an administrator account, Windows OpenSSH uses a different file:

```powershell
notepad "C:\ProgramData\ssh\administrators_authorized_keys"
icacls.exe "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
Restart-Service sshd
```

Prefer the standard user path unless admin access is truly required.

## Alex: Local Verification

Run these from Alex's machine:

```powershell
Get-Service sshd
Test-NetConnection -ComputerName 127.0.0.1 -Port 22
& "$env:ProgramFiles\Tailscale\tailscale.exe" ip -4
```

Expected:

- `sshd` is `Running`
- Local port 22 test succeeds
- Tailscale prints a `100.x.y.z` IPv4 address

## Ibrahim: First SSH Test

From Ibrahim's machine:

```bash
tailscale status
ssh -4 AlexWindowsUser@100.x.y.z
```

Replace:

- `AlexWindowsUser` with the Windows username Alex provided
- `100.x.y.z` with Alex's Tailscale IPv4 address

If MagicDNS is enabled and the name resolves:

```bash
ssh -4 AlexWindowsUser@alex-windows
```

The first connection may ask to trust the host key. Verify the target is Alex's Tailscale address before accepting.

## Optional: VS Code Remote SSH

After raw SSH works, configure VS Code Remote SSH on Ibrahim's machine:

```sshconfig
Host alex-windows
    HostName 100.x.y.z
    User AlexWindowsUser
    AddressFamily inet
    IdentityFile ~/.ssh/id_ed25519
```

Then use VS Code's **Remote-SSH: Connect to Host** command and select `alex-windows`.

## Optional: Tighten Tailnet Access Controls

The default tailnet policy commonly allows broad device-to-device access. For a safer setup, restrict Ibrahim's account to TCP/22 on Alex's device.

After Alex's PC is visible in the admin console, record its Tailscale IPv4 address and add a host alias in the tailnet policy file:

```jsonc
{
  "hosts": {
    "alex-windows": "100.x.y.z"
  },
  "grants": [
    {
      "src": ["iseye@mearas.us"],
      "dst": ["alex-windows"],
      "ip": ["tcp:22"]
    }
  ]
}
```

If the policy already has other `grants`, merge this entry into the existing file instead of replacing everything. Validate the policy in the Tailscale admin console before saving.

## Troubleshooting

If `ssh` times out:

- Confirm Alex's Tailscale status is connected.
- Confirm Ibrahim can see Alex's device in `tailscale status`.
- Confirm Alex's firewall rule allows `100.64.0.0/10` to TCP/22.
- Use the direct Tailscale IPv4 address first, not MagicDNS.

If `ssh` says permission denied:

- Confirm the SSH username is the Windows local username, not the display name.
- Confirm Ibrahim's public key is in the correct `authorized_keys` file.
- If the target Windows account is an administrator, use `C:\ProgramData\ssh\administrators_authorized_keys`.
- Confirm file permissions were set with `icacls`.

If MagicDNS does not work:

- Use `ssh -4 AlexWindowsUser@100.x.y.z`.
- Confirm MagicDNS is enabled in the Tailscale DNS page.
- Check the machine name in the Tailscale Machines page.

## Official References

- Tailscale Windows install: https://tailscale.com/docs/install/windows
- Tailscale invite external users: https://tailscale.com/docs/features/sharing/how-to/invite-any-user
- Tailscale SSH limitations and server platform support: https://tailscale.com/docs/features/tailscale-ssh
- SSH over Tailscale with a normal SSH server: https://tailscale.com/docs/reference/ssh-over-tailscale
- Tailscale MagicDNS: https://tailscale.com/docs/features/magicdns
- Tailscale firewall guidance: https://tailscale.com/docs/reference/faq/firewall-ports
- Tailscale reserved IP ranges: https://tailscale.com/docs/reference/reserved-ip-addresses
- Tailscale grants and ACLs: https://tailscale.com/docs/reference/syntax/policy-file
- Microsoft OpenSSH Server install: https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse
- Microsoft OpenSSH key-based authentication: https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement
- Microsoft OpenSSH Server configuration: https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration
