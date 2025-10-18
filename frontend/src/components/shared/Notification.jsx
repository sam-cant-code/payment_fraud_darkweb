import React, { useEffect, useState } from 'react';
import { CheckCircle, AlertTriangle, X } from 'lucide-react';

const Notification = ({ message, type, onClose }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (message) {
      setIsVisible(true);
    } else {
      setIsVisible(false);
    }
  }, [message]);

  if (!isVisible) return null;

  const isError = type === 'error';
  const bgColor = isError ? 'bg-status-deny-bg' : 'bg-status-approve-bg';
  const textColor = isError ? 'text-status-deny-text' : 'text-status-approve-text';
  const Icon = isError ? AlertTriangle : CheckCircle;

  return (
    <div
      className={`fixed top-5 right-5 z-50 min-w-[300px] rounded-lg shadow-xl p-4 flex items-start gap-3 ${bgColor} ${textColor} animate-fade-in-right`}
    >
      <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" />
      <div className="flex-1">
        <h4 className="font-semibold">{isError ? 'Error' : 'Success'}</h4>
        <p className="text-sm">{message}</p>
      </div>
      <button onClick={onClose} className="text-current opacity-70 hover:opacity-100">
        <X className="w-5 h-5" />
      </button>
    </div>
  );
};

export default Notification;