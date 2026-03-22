export interface TripletexCredentials {
  base_url: string;
  session_token: string;
}

export interface FileAttachment {
  filename: string;
  content_base64: string; // base64-encoded (field name per competition spec)
  mime_type: string;
}

export interface SolveRequestBody {
  prompt: string;
  files?: FileAttachment[];
  tripletex_credentials: TripletexCredentials;
  use_sandbox?: boolean;
}
