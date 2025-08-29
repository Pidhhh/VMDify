import React, { useEffect, useRef } from 'react';

const VideoPreview = ({ frameB64, format = 'jpeg' }) => {
  const imgRef = useRef(null);

  useEffect(() => {
    // nothing to do; src is bound
  }, [frameB64]);

  if (!frameB64) {
    return (
      <div className="card-dark p-3 text-sm text-gray-400">
        Waiting for frames...
      </div>
    );
  }

  return (
    <div className="rounded-lg overflow-hidden border border-gray-700 bg-black">
      <img
        ref={imgRef}
        alt="Detection Preview"
        src={`data:image/${format};base64,${frameB64}`}
        style={{ display: 'block', width: '100%' }}
      />
    </div>
  );
};

export default VideoPreview;
