# Deploy bang Docker (thay systemd)

Tai lieu nay giu nguyen logic app hien tai, chi thay doi cach chay service: tu systemd sang Docker.

## 1) Build va chay container

Chay trong thu muc du an:

```bash
docker compose up -d --build
```

Kiem tra trang thai:

```bash
docker compose ps
docker compose logs -f whisper-api
```

App se lang nghe tai `127.0.0.1:8000` tren host.

## 2) Nginx (giu nhu cu)

Neu ban dang dung Nginx tren host, chi can proxy ve `127.0.0.1:8000` nhu truoc.

Vi du block `location`:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    client_max_body_size 1000m;
}
```

Sau khi sua Nginx:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

## 3) Tat systemd cu (neu dang chay)

Thay `whisper-api.service` bang ten service that cua ban:

```bash
sudo systemctl stop whisper-api.service
sudo systemctl disable whisper-api.service
```

Neu muon chan service tu khoi dong lai:

```bash
sudo systemctl mask whisper-api.service
```

## 4) Lenh van hanh thay cho systemd

Restart app:

```bash
docker compose restart whisper-api
```

Update code + redeploy:

```bash
git pull
docker compose up -d --build
```

Xem logs:

```bash
docker compose logs -f whisper-api
```

Dung dich vu:

```bash
docker compose down
```

## 5) Du lieu duoc giu lai

- Thu muc output `file_vtt` duoc mount tu host vao container.
- Thu muc `models` duoc mount de tai su dung model local.
- Whisper cache duoc luu qua Docker volume `whisper-cache`.

## 6) Google Drive credentials

De dung link Google Drive (file/folder), dat file `credentials.json` o thu muc goc du an.

Ho tro 2 kieu:

- Service account JSON (khuyen nghi).
- JSON co truong `api_key` (chi dung duoc voi resource public).

Neu khong dung file, co the set bien moi truong `GOOGLE_DRIVE_API_KEY`.

Convert 1 file Drive bằng endpoint cũ /convert
Body JSON:
{
"video_url": "https://drive.google.com/file/d/FILE_ID/view?usp=sharing",
"language": "vi"
}

Convert cả folder Drive bằng endpoint mới /convert-drive-folder
Body JSON:
{
"folder_url": "https://drive.google.com/drive/folders/1g-ZDze6jVI_Y418_O7vHh7QPqI0JXOjq",
"language": "vi"
}
