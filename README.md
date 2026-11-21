# File Tools MCP + Cloudflare Tunnel

Gives Claude sandboxed file tools: list_directory, read_file, read_multiple_files, file_info,
find_files, search_text, create_file, write_file, append_file, edit_file, edit_file_multi,
replace_lines, insert_lines, make_directory, move_path, copy_path, delete_path, get_server_info.

## Setup (once)
1. Install Python 3.10+ and cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and set `MCP_ROOT` and `MCP_DOMAIN`
   (the domain must be on your Cloudflare account)

# Install cloudfare   
```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared
```


## Run (every time)
    python run.py

First run: a browser opens for `cloudflared tunnel login`; after that it is fully automatic
(creates/reuses the tunnel, sets the DNS record, starts server + tunnel, restarts them if they crash).

It prints a URL like `https://mcp.yourdomain.com/<secret>/mcp`.

## Connect to Claude
Settings -> Connectors -> Add custom connector -> paste that URL (no OAuth fields needed).

## Security
- Access is limited to MCP_ROOT (symlink escapes are blocked).
- The secret in the URL is the only authentication. Treat it like a password; rotate by changing MCP_SECRET.
- Use MCP_READ_ONLY=1 to disable all write tools.
- Stronger option: put Cloudflare Access in front (note Claude's connector must still be able to reach it).

## Troubleshooting
- `curl https://mcp.yourdomain.com/healthz` should print `ok`.
- 404 in Claude = wrong/old secret in the URL.
- Local test without Cloudflare: `SKIP_TUNNEL=1 python run.py`, then use the printed http URL with MCP Inspector (`npx @modelcontextprotocol/inspector`).
