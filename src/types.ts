export interface TripletexCredentials {
  proxy_url: string;
  session_token: string;
}

export interface FileAttachment {
  filename: string;
  content: string; // base64-encoded
  mime_type: string;
}

export interface SolveRequestBody {
  prompt: string;
  files?: FileAttachment[];
  tripletex_credentials: TripletexCredentials;
}
